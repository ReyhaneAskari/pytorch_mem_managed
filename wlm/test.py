import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torch.nn.functional as F
from torch.autograd import Variable

import unittest
import time
import sys

import word_language_model as wlm_baseline


class TestMemoryBaseline(unittest.TestCase):
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

    def test_wlm_baseline(self):
        total_iters = 20
        iterations = 5

        model_name = 'LSTM'
        ntokens = 33278
        emsize = 200
        nhid = 200
        nlayers = 1
        dropout = 0.2
        tied = False
        batchsize = 250
        bptt = 40

        data = Variable(torch.LongTensor(bptt, batchsize).fill_(1), volatile=False)
        target_var = Variable(torch.LongTensor(bptt * batchsize).fill_(1))
        targets = target_var.cuda()
        input_data = data.cuda()

        model = wlm_baseline.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied)
        model = model.cuda()
        model.train()
        criterion = nn.CrossEntropyLoss().cuda()
        hidden = model.init_hidden(batchsize)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        total_gpu_time = []
        total_cpu_time = []
        with cudnn.flags(enabled=True, benchmark=True):
            for i in range(total_iters):
                start.record()
                start_cpu = time.time()
                for j in range(iterations):
                    hidden = self.repackage_hidden(hidden)
                    output, hidden = model(input_data, hidden)
                    loss = criterion(output.view(-1, ntokens), targets)
                    loss = loss + 0*(hidden[0].sum() + hidden[1].sum())
                    loss.backward()

                end_cpu = time.time()
                end.record()
                torch.cuda.synchronize()
                gpu_msec = start.elapsed_time(end)
                total_gpu_time.append(gpu_msec * 1000)
                total_cpu_time.append((end_cpu - start_cpu) * 1000000)
                print("Baseline WLM ({:2d}): ({:8.3f} usecs gpu) ({:8.3f} usecs cpu)".format(
                    i, gpu_msec * 1000, (end_cpu - start_cpu) * 1000000,
                    file=sys.stderr))
        print("avg gpu time: " + str((sum(total_gpu_time[1:]) / len(total_gpu_time[1:]))/1000000))
        print("total cpu time: " + str((sum(total_cpu_time[1:]))/1000000))


if __name__ == '__main__':
    unittest.main()

