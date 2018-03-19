import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest
import time
import sys
import resnet as resnet_baseline



class TestMemoryBaseline(unittest.TestCase):

    def test_resnet_baseline(self):
        # N = 8
        # total_iters = 10    # (warmup + benchmark)
        # iterations = 4

        N = 32
        total_iters = 10    # (warmup + benchmark)
        iterations = 1

        target = Variable(torch.randn(N).fill_(1)).type("torch.LongTensor")
        x = Variable(torch.randn(N, 3, 32, 32).fill_(1.0), requires_grad=True)
        model = resnet_baseline.resnet1001()

        # switch the model to train mode
        model.train()

        # convert the model and input to cuda
        model = model.cuda()
        input_var = x.cuda()
        target_var = target.cuda()

        # declare the optimizer and criterion
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD(model.parameters(), 0.01, momentum=0.9, weight_decay=1e-4)
        optimizer.zero_grad()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_gpu_time = []
        total_cpu_time = []
        with cudnn.flags(enabled=True, benchmark=True):
            for i in range(total_iters):
                start.record()
                start_cpu = time.time()
                for j in range(iterations):
                    output = model(input_var)
                    loss = criterion(output, target_var)
                    loss.backward()
                    optimizer.step()

                end_cpu = time.time()
                end.record()
                torch.cuda.synchronize()
                gpu_msec = start.elapsed_time(end)
                total_gpu_time.append(gpu_msec * 1000)
                total_cpu_time.append((end_cpu - start_cpu) * 1000000)
                print("Baseline resnet ({:2d}): ({:8.3f} usecs gpu) ({:8.3f} usecs cpu)".format(
                    i, gpu_msec * 1000, (end_cpu - start_cpu) * 1000000,
                    file=sys.stderr))
        print("avg gpu time: " + str(sum(total_gpu_time[1:]) / len(total_gpu_time[1:])))
        print("avg cpu time: " + str(sum(total_cpu_time[1:]) / len(total_cpu_time[1:])))
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

if __name__ == '__main__':
    unittest.main()

