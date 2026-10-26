#include <torch/torch.h>
#include <iostream>

torch::Tensor forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V);
int main()
{
    const int batch_size = 2;
    const int n_head = 8;
    const int seq_len = 256;
    const int head_embd = 128;

    auto q = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    auto k = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    auto v = torch::rand({batch_size, n_head, seq_len, head_embd}).cuda();
    forward(q, k, v);
    return 0;
}