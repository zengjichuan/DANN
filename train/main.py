import random
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np
from test import test

source_train = os.path.join('data', "skype", "skype_forum.train.json")
source_eval = os.path.join('data', "skype", "skype_forum.eval.json")
target_train = os.path.join('data', 'skype', "skype_review.json")
embedding_file = ""
model_root = os.path.join('models')
cuda = True
cudnn.benchmark = True
lr = 1e-3
batch_size = 64
n_epoch = 100

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

####################################################################################
#  Load data
####################################################################################
dataset_source_train = GetLoader(data_fn=source_train,)
dataset_source_eval = GetLoader(data_fn=source_eval,)
dataset_target_train = GetLoader(data_fn=target_train,)

dataloader_source_train = torch.utils.data.DataLoader(dataset=dataset_source_train, batch_size=batch_size, shuffle=True)
dataloader_source_eval = torch.utils.data.DataLoader(dataset=dataset_source_eval, batch_size=batch_size, shuffle=False)
dataloader_target_train = torch.utils.data.DataLoader(dataset=dataset_target_train, batch_size=batch_size, shuffle=True)

####################################################################################
#  Load model
####################################################################################
my_net = CNNModel()


####################################################################################
#  Setup optimizer
####################################################################################

optimizer = optim.Adam(my_net.parameters(), lr=lr)

loss_class = torch.nn.NLLLoss()
loss_domain = torch.nn.NLLLoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True

# training

for epoch in xrange(n_epoch):

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    data_source_iter = iter(dataloader_source)
    data_target_iter = iter(dataloader_target)

    i = 0
    while i < len_dataloader:

        p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
        alpha = 2. / (1. + np.exp(-10 * p)) - 1

        # training model using source data
        data_source = data_source_iter.next()
        s_img, s_label = data_source

        my_net.zero_grad()
        batch_size = len(s_label)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        domain_label = torch.zeros(batch_size)
        domain_label = domain_label.long()

        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(s_img).copy_(s_img)
        class_label.resize_as_(s_label).copy_(s_label)
        inputv_img = Variable(input_img)
        classv_label = Variable(class_label)
        domainv_label = Variable(domain_label)

        class_output, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_s_label = loss_class(class_output, classv_label)
        err_s_domain = loss_domain(domain_output, domainv_label)

        # training model using target data
        data_target = data_target_iter.next()
        t_img, _ = data_target

        batch_size = len(t_img)

        input_img = torch.FloatTensor(batch_size, 3, image_size, image_size)
        domain_label = torch.ones(batch_size)
        domain_label = domain_label.long()

        if cuda:
            t_img = t_img.cuda()
            input_img = input_img.cuda()
            domain_label = domain_label.cuda()

        input_img.resize_as_(t_img).copy_(t_img)
        inputv_img = Variable(input_img)
        domainv_label = Variable(domain_label)

        _, domain_output = my_net(input_data=inputv_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domainv_label)
        err = err_t_domain + err_s_domain + err_s_label
        err.backward()
        optimizer.step()

        i += 1

        print 'epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
              % (epoch, i, len_dataloader, err_s_label.cpu().data.numpy(),
                 err_s_domain.cpu().data.numpy(), err_t_domain.cpu().data.numpy())

    torch.save(my_net, '{0}/mnist_mnistm_model_epoch_{1}.pth'.format(model_root, epoch))
    test(source_dataset_name, epoch)
    test(target_dataset_name, epoch)

print 'done'
