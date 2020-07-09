import torch
import torchvision
import torchvision.transforms as transforms

from full_body_net import FullBodyNet
from mars_dataset import Mars
from triplet_loss import triplet_loss, total_triplets_valid, invalid_triplet_indices

from queue import Queue


# transform for nets trained on imagenet
def imagenet_transform():
    return transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# train all inception layers
def full_body_net_optimiser(net, lr, momentum):
    return torch.optim.SGD([
        {'params':net.inception_v3.fc.parameters(),       'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_7a.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_7b.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_7c.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_6a.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_6b.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_6c.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_6d.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_6e.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_5b.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_5c.parameters(), 'lr':lr, 'momentum':momentum},
        {'params':net.inception_v3.Mixed_5d.parameters(), 'lr':lr, 'momentum':momentum},
        ])


def train(max_iters, batch_size, minibatch_size, lr, momentum):

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


    # show some examples
    import matplotlib.pyplot as plt
    import numpy as np

    # imshow
    def imshow(img):
        img = img / 2 + 0.5 # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    # imshow first batch
    dataiter = iter(trainloader)
    anc, pos, neg = dataiter.next()[0]
    imshow(torchvision.utils.make_grid(anc))


    # inception_v3
    # resize final layer for classes above
    net = FullBodyNet()
    net.to(device)
    net.train()


    # optimiser for training
    optimiser = full_body_net_optimiser(net, lr, momentum)
    optimiser.zero_grad()


    # loop variables
    iteration = 0
    running_loss = 0.0

    minibatch_queue = Queue()
    mbatches_processed = 0

    
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
                mbatches_processed +=1 

                # optimise at batch size
                if mbatches_processed >= batch_size/minibatch_size:
                    optimiser.step()
                    optimiser.zero_grad()
                
                    # next iteration
                    running_loss += loss.item()
                    mbatches_processed = 0
                    iteration += 1
                    
                    # print loss every 10 iters
                    if iteration % 10 == 0:
                        print('[%d] loss: %.6f' % (iteration, running_loss / 10 * batch_size))
                        running_loss = 0.0

                    # validate every 100 iters
                    if iteration % 100 == 0:
                        testloader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=True, num_workers=2)
                        no_test = 1000
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
                        
                        print('Accuracy on {} random test triplets: {}'.format(no_test, 100 * correct / total))

                    # save model every 20000 iters
                    if iteration % 20000 == 0:
                        torch.save(net.state_dict(), './mars_triplet_{}.pth'.format(iteration))

    # save model
    torch.save(net.state_dict(), './mars_triplet.pth')
    print('Finished Training')

if __name__ == '__main__':
    # hyperparams
    max_iters = 120000
    batch_size = 64
    minibatch_size = 8
    lr = 0.001
    momentum = 0.9

    # train
    train(max_iters, batch_size, minibatch_size, lr, momentum)