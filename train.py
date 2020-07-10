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


# train all inception layers
def full_body_net_optimiser(net, lr, momentum):
    # return torch.optim.SGD([
    #     {'params':net.inception_v3.fc.parameters(),       'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_7a.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_7b.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_7c.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_6a.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_6b.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_6c.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_6d.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_6e.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_5b.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_5c.parameters(), 'lr':lr, 'momentum':momentum},
    #     {'params':net.inception_v3.Mixed_5d.parameters(), 'lr':lr, 'momentum':momentum},
    #     ])
    return torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

def train(max_iters, batch_size, minibatch_size, lr, momentum, statepath=None, logpath=None):

    # use GPU if avail
    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'
    

    # load dataset
    transform = imagenet_transform()
    trainset = Mars(root='./.data', train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=2)
    testset = Mars(root='./.data', train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=True, num_workers=2)


    # inception_v3
    # resize final layer for classes above
    net = FullBodyNet()
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
    else:
        print('Loading FullBodyNet metadata from {}'.format(logpath))
        with open(logpath, 'r') as f:
            iteration = int(f.readline())
    
    running_loss = 0.0
    minibatch_queue = Queue()
    mbatches_processed = 0

    val_history   = []
    train_history = []
    loss_history  = []
    
    # train the thing
    while iteration < max_iters:
        for data in trainloader:

            # ONLINE TRIPLET SELECTION
            # get all of the semi-hard triplets here to make a minibatch
            with torch.no_grad():
                anc, pos, neg = data[0]
                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                anc_outputs, pos_outputs, neg_outputs = net(anc)[0], net(pos)[0], net(neg)[0]

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
                anc_outputs, pos_outputs, neg_outputs = net(anc)[0], net(pos)[0], net(neg)[0]

                # forward + backward
                loss = triplet_loss(anc_outputs, pos_outputs, neg_outputs)
                loss.backward()
                running_loss += loss.item()
                mbatches_processed +=1 

                # output every tenth of a batch
                if mbatches_processed % int((batch_size/minibatch_size)/10) == 0:
                    print('\taccumulated loss: %.6f' % (running_loss / (mbatches_processed * minibatch_size)))
                    show_triplet('invalid triplet', anc[0], pos[0], neg[0])

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
                    
                    # validate every 100 iters
                    if iteration % 10 == 0:
                        no_test = 500
                        correct = 0
                        total = 0

                        with torch.no_grad():
                            for j, data in enumerate(testloader):
                                anc, pos, neg = data[0]
                                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                                anc_outputs, pos_outputs, neg_outputs = net(anc)[0], net(pos)[0], net(neg)[0]

                                correct += total_triplets_valid(anc_outputs, pos_outputs, neg_outputs).item()
                                total += anc.size(0)

                                if j == int(no_test/minibatch_size):
                                    break
                        
                        val_accuracy = 100 * correct / total
                        print('Accuracy on first {} validation triplets: {}'.format(no_test, val_accuracy))

                        correct = 0
                        total = 0

                        with torch.no_grad():
                            for j, data in enumerate(trainloader):
                                anc, pos, neg = data[0]
                                anc, pos, neg = anc.to(device), pos.to(device), neg.to(device)
                                anc_outputs, pos_outputs, neg_outputs = net(anc)[0], net(pos)[0], net(neg)[0]

                                correct += total_triplets_valid(anc_outputs, pos_outputs, neg_outputs).item()
                                total += anc.size(0)

                                if j == int(no_test/minibatch_size):
                                    break
                        
                        train_accuracy = 100 * correct / total
                        print('Accuracy on first {} train triplets: {}'.format(no_test, train_accuracy))

                    # add to history for log
                    if iteration % 200 == 0:
                        val_history.append(val_accuracy)
                        train_history.append(train_accuracy)
                        loss_history.append(current_loss)
                    
                    # checkpoint model every 40 iterations
                    # (approx 1h at minibatch_size=8)
                    if iteration % 40 == 0:
                        torch.save(net.state_dict(), './mars_triplet_ckpt.pth'.format(iteration))
                        with open('./mars_triplet_ckpt.log', 'w') as f:
                            f.write(str(iteration))
                            f.write('\n')
                            f.write(str(val_history))
                            f.write('\n')
                            f.write(str(train_history))
                            f.write('\n')
                            f.write(str(loss_history))
                            f.write('\n')

                        print('Model saved to ./mars_triplet_ckpt.pth')
                        print('Metadata saved to ./mars_triplet_ckpt.log')
                    
                    # save model every 500 iters
                    if iteration % 500 == 0:
                        torch.save(net.state_dict(), './mars_triplet_{}.pth'.format(iteration))
                        print('Model saved to ./mars_triplet_{}.pth'.format(iteration))

    # save model
    torch.save(net.state_dict(), './mars_triplet.pth')
    print('Model saved to ./mars_triplet.pth')
    print('Finished Training')

if __name__ == '__main__':
    # hyperparams
    max_iters = 100000
    batch_size = 1024
    minibatch_size = 8
    lr = 0.001
    momentum = 0.9

    # train
    train(max_iters, batch_size, minibatch_size, lr, momentum, './mars_triplet_ckpt.pth', './mars_triplet_ckpt.log')