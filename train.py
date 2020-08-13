import torch
import torchvision
import torchvision.transforms as transforms

from full_body_net import FullBodyNet
from mars_dataset import Mars
from triplet_loss import triplet_loss, total_triplets_valid, invalid_triplet_indices

import cv2
from queue import Queue


# use cv2 to show a triplet
def show_triplet(text, anc, pos, neg):
    anc, pos, neg = anc.cpu(), pos.cpu(), neg.cpu()
    img = torchvision.utils.make_grid(torch.stack([anc, pos, neg])).numpy()
    img = img.transpose((1,2,0))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 2 + 0.5
    cv2.imshow(text, img)
    cv2.waitKey(1)


# transform for nets trained on imagenet
def imagenet_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# train the final residual
def full_body_net_optimiser(net, lr, momentum):
    return torch.optim.SGD([
        {'params':net.resnet50.fc.parameters(),       'lr':lr, 'momentum':momentum},
        {'params':net.resnet50.layer4.parameters(), 'lr':lr, 'momentum':momentum},
        ])

def train(max_iters, batch_size, minibatch_size, images_per_person, lr, momentum, statepath=None, logpath=None):

    # use GPU if avail
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'

    # load dataset
    persons_per_batch = max(1, int(minibatch_size/images_per_person))
    transform = imagenet_transform()
    trainset = Mars(root='./.data', train=True, transform=transform, triplets_per_image=16)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=persons_per_batch, shuffle=True, num_workers=2)
    testset = Mars(root='./.data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=persons_per_batch, shuffle=True, num_workers=2)


    # inception_v3
    # resize final layer for classes above
    net = FullBodyNet(train=True)
    if statepath is not None:
        print('Loading FullBodyNet state from {}'.format(statepath))
        net.load_state_dict(torch.load(statepath))
    net.to(device)
    net.train()


    # optimiser for training
    optimiser = full_body_net_optimiser(net, lr, momentum)
    optimiser.zero_grad()


    # loop variables
    if logpath is None:
        iteration = 0
        with open('./mars_triplet_ckpt.log', 'w') as f:
            f.write('')
    else:
        print('Loading FullBodyNet metadata from {}'.format(logpath))
        with open(logpath, 'r') as f:
            while 1:
                line = f.readline()
                if line == '':
                    break
                
                iteration = int(line)
                f.readline()
                f.readline()
                f.readline()
    
    running_loss = 0.0
    minibatch_queue = Queue()
    mbatches_processed = 0

    # train the thing
    while iteration < max_iters:
        for data in trainloader:

            # ONLINE TRIPLET SELECTION
            # get all of the semi-hard triplets here to make a minibatch
            with torch.no_grad():
                for person in data[0]:
                    anc, pos, neg = person
                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                    anc_outputs, pos_outputs, neg_outputs = net(anc), net(pos), net(neg)

                    # add triplets that violate the loss fn to the minibatch
                    indices = invalid_triplet_indices(anc_outputs, pos_outputs, neg_outputs).flatten().tolist()
                    for index in indices:
                        minibatch_queue.put((anc[index], pos[index], neg[index]))

                # correct = total_triplets_valid(anc_outputs, pos_outputs, neg_outputs).item()
                # print (minibatch_size - correct, minibatch_queue.qsize())

            # FORWARD MINIBATCH
            if minibatch_queue.qsize() >= minibatch_size:
                anc, pos, neg = [], [], []
                
                for j in range(minibatch_size):
                    anc_img, pos_img, neg_img = minibatch_queue.get()
                    anc.append(anc_img)
                    pos.append(pos_img)
                    neg.append(neg_img)

                anc, pos, neg = torch.stack(anc), torch.stack(pos), torch.stack(neg)
                anc_outputs, pos_outputs, neg_outputs = net(anc), net(pos), net(neg)

                # forward + backward
                loss = triplet_loss(anc_outputs, pos_outputs, neg_outputs)
                loss.backward()
                running_loss += loss.item()
                mbatches_processed +=1

                # output every tenth of a batch
                if mbatches_processed % max(1, int((batch_size/minibatch_size)/10)) == 0:
                    print('\taccumulated loss: %.6f' % (running_loss / (mbatches_processed * minibatch_size)))

                # optimise at batch size
                if mbatches_processed >= batch_size/minibatch_size:
                    optimiser.step()
                    optimiser.zero_grad()
                    current_loss = running_loss/batch_size
                
                    # next iteration
                    mbatches_processed = 0
                    iteration += 1
                    running_loss = 0.0

                    # print current loss
                    print('[%d] batch loss: %.6f' % (iteration, current_loss))
                    show_triplet('invalid triplet', anc[0], pos[0], neg[0])
                    
                    # validate every 50 iters
                    if iteration % 50 == 0:
                        no_test = 1000
                        correct = 0.0
                        total = 0.0

                        with torch.no_grad():
                            for j, data in enumerate(testloader):
                                for person in data[0]:
                                    anc, pos, neg = person
                                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                                    anc_outputs, pos_outputs, neg_outputs = net(anc), net(pos), net(neg)

                                    correct += total_triplets_valid(anc_outputs, pos_outputs, neg_outputs).item()
                                    total += anc.size(0)

                                if j == int(no_test/minibatch_size):
                                    break
                        
                        val_accuracy = 100 * correct / total
                        print('Accuracy on first {} validation triplets: {}'.format(no_test, val_accuracy))

                        correct = 0.0
                        total = 0.0

                        with torch.no_grad():
                            for j, data in enumerate(trainloader):
                                for person in data[0]:
                                    anc, pos, neg = person
                                    anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                                    anc_outputs, pos_outputs, neg_outputs = net(anc), net(pos), net(neg)

                                    correct += total_triplets_valid(anc_outputs, pos_outputs, neg_outputs).item()
                                    total += anc.size(0)

                                if j == int(no_test/minibatch_size):
                                    break
                        
                        train_accuracy = 100 * correct / total
                        print('Accuracy on first {} train triplets: {}'.format(no_test, train_accuracy))
                    
                    # checkpoint model every 100 iterations
                    if iteration > 500 and iteration % 100 == 0:
                        torch.save(net.state_dict(), './mars_triplet_ckpt.pth'.format(iteration))
                        print('Model saved to ./mars_triplet_ckpt.pth')

                        with open('./mars_triplet_ckpt.log', 'w') as f:
                            f.write(str(iteration) + '\n')
                            f.write(str(val_accuracy) + '\n')
                            f.write(str(train_accuracy) + '\n')
                            f.write(str(current_loss) + '\n')
                        print('Metadata saved to ./mars_triplet_ckpt.log')
                    
                    # save model every 500 iters
                    if iteration % 500 == 0:
                        torch.save(net.state_dict(), './mars_triplet_{}.pth'.format(iteration))
                        print('Model saved to ./mars_triplet_{}.pth'.format(iteration))
                    
                    if iteration >= max_iters:
                        break

    print('Finished Training')

if __name__ == '__main__':
    # hyperparams
    max_iters = 1000
    batch_size = 72
    minibatch_size = 8
    images_per_person = 4
    lr = 0.001
    momentum = 0.9

    # train
    train(max_iters, batch_size, minibatch_size, images_per_person, lr, momentum, './mars_triplet_ckpt.pth', './mars_triplet_ckpt.log')
    # train(max_iters, batch_size, minibatch_size, images_per_person, lr, momentum)