Credit goes to: https://github.com/prigoyal/pytorch_memonger

This repo contains the tests for using cpu as swap for gpu in pytorch. The changes in pytorch are in this branch: https://github.com/ReyhaneAskari/pytorch/tree/manage_cpu_ram


Benchmark results:



|                                | RESNET             | RESNET managed     | RESNET with only general change | RESNET with only alloc change | RESNET managed      | RESNET managed      | RESNET with only alloc change |
|--------------------------------|--------------------|--------------------|---------------------------------|-------------------------------|---------------------|---------------------|-------------------------------|
| Total iterations (runs * iter) | 4 * 10             | 4 * 10             | 4 * 10                          | 4 * 10                        | 2 * 10              | 1 * 10              | 1 * 10                        |
| mini-batch-size                | 8                  | 8                  | 8                               | 8                             | 16                  | 32                  | 32                            |
| avg gpu time over 10 iters     | 1.581939 s         | 8.094114 s         | 1.932749 s                      | 8.239344 s                    | 5.041900 s          | 7.056532 s          | 7.092366 s                    |
| total time except 1st run      | 14.2369 s          | 72.846 s           | 17.394                          | 74.152 s                      | 45.105              | 62.662 s            | 63.009 s                      |
| total time                     | 27.171s            | 88.398s            | 28.577s                         | 90.398s                       | 58.478s             | 78.065s             | 78.618s                       |
| Data on GPU                    | 8120MiB / 12066MiB | 8172MiB / 12066MiB | 8140MiB / 12066MiB              | 8158MiB / 12066MiB            | 12064MiB / 12066MiB | 12064MiB / 12066MiB | 12064MiB / 12066MiB           |




|                                | VNET               | VNET managed       | VNET managed        | VNET managed        | VNET with only alloc change |
|--------------------------------|--------------------|--------------------|---------------------|---------------------|-----------------------------|
| Total iterations (runs * iter) | 4 * 10             | 4 * 10             | 2 * 10              | 1 * 10              | 1 * 10                      |
| mini-batch-size                | 4                  | 4                  | 8                   | 16                  | 16                          |
| avg gpu time over 10 iters     | 10.536860 s        | 11.024819 s        | 15.294250 s         | 17.153561 s         | 17.153326 s                 |
| total time except 1st run      | 94.8305            | 99.222 s           | 137.144 s           | 136.810 s           | 136.74863 s                 |
| total time                     | 121.88827 s        | 130.893s           | 178.042 s           | 216.938 s           | 216.445                     |
| Data on GPU                    | 9452MiB / 12066MiB | 8622MiB / 12066MiB | 12064MiB / 12066MiB | 12064MiB / 12066MiB | 12064MiB / 12066MiB         |
