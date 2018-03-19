import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest
import time
import sys

import vnet as vnet_baseline


class TestMemoryBaseline(unittest.TestCase):

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv3d') != -1:
            nn.init.kaiming_normal(m.weight)
            m.bias.data.zero_()

    def test_vnet_baseline(self):
        # N = 4
        # total_iters = 10
        # iterations = 2

        N = 4
        total_iters = 10    # (warmup + benchmark)
        iterations = 4

        total_start = time.time()
        target = Variable(torch.randn(N, 1, 128, 128, 64).fill_(1)).type("torch.LongTensor")
        x = Variable(torch.randn(N, 1, 128, 128, 64).fill_(1.0), requires_grad=True)
        model = vnet_baseline.VNet(elu=False, nll=True)
        bg_weight = 0.5
        fg_weight = 0.5
        weights = torch.FloatTensor([bg_weight, fg_weight])
        weights = weights.cuda()
        model.train()

        model = model.cuda()
        input_var = x.cuda()
        target_var = target.cuda()
        target = target_var.view(target_var.numel())

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=0.99, weight_decay=1e-8)
        optimizer.zero_grad()
        model.apply(self.weights_init)
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
                    loss = F.nll_loss(output, target, weight=weights)
                    loss.backward()
                    optimizer.step()
                    if j == 0:
                        continue
                if j == 0 and i == 0:
                    print("skipped first batch")
                    continue
                end_cpu = time.time()
                end.record()
                torch.cuda.synchronize()
                gpu_msec = start.elapsed_time(end)
                total_gpu_time.append(gpu_msec * 1000)
                total_cpu_time.append((end_cpu - start_cpu) * 1000000)
                print("Baseline vnet ({:2d}): ({:8.3f} usecs gpu) ({:8.3f} usecs cpu)".format(
                    i, gpu_msec * 1000, (end_cpu - start_cpu) * 1000000,
                    file=sys.stderr))
        print("total_time: " + str(time.time() - total_start))
        print("avg gpu time: " + str(sum(total_gpu_time[1:]) / len(total_gpu_time[1:])))
        print("total cpu time: " + str((sum(total_cpu_time[1:]) / 1000000)))

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

if __name__ == '__main__':
    unittest.main()

