# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# Scaled down for demo purposes. Recommend training with 2-4 GPUs.

from utils import get_data_loader, get_data_loader_train, pretraining_loss, get_sequential_model, set_random_seed
import argparse
from hydra import ModelOrchestrator, ModelTask
import torch
from deepspeed.profiling.flops_profiler import FlopsProfiler


def run_test(model):
    model.cpu()
    valid_loader = get_data_loader(2)
    ctr = 0
    with torch.no_grad():
        accum_loss = 0
        for sample, label in valid_loader:
            ctr+=1
            outputs = new_model(sample)
            my_loss = pretraining_loss(outputs, label)
            accum_loss += my_loss.item()
            if (ctr % 10 == 0):
                print("SAMPLE: {} / {}".format(ctr, len(valid_loader)))
                print("RUNNING PPL: {}".format(math.exp(accum_loss/ctr)))
    print("TEST LOSS: {}".format(math.exp(accum_loss/ctr)))
    return math.exp(accum_loss/ctr)

def report_flops(model, sample):
    macs, params = get_model_profile(model=model, input_res=sample.shape)
    print("PARAMETERS: {}".format(params))
    return 2 * macs / (1000 ** 3)
            
def main(seed):
    set_random_seed(seed)
    all_tasks = []
    all_models = []
    all_dataloaders = []
    lr_names = ["3e-4", "1e-4", "5e-5"]
    
    learning_rates = [3e-4, 1e-4, 5e-5]
    learning_rates=[3e-4, 1e-4]
    batch_sizes = [1, 2, 4, 8]
    batch_sizes=[4]
    profilers = []
    for idx, lr in enumerate(learning_rates):
        for b_size in batch_sizes:
            dataloader = get_data_loader(b_size)
            all_dataloaders.append(dataloader)
            new_model = get_sequential_model()
            all_models.append(new_model)
            profilers.append(FlopsProfiler(new_model))
            task = ModelTask("MODEL_{}_{}".format(lr_names[idx], b_size), new_model, pretraining_loss, dataloader, lr, 1)
            all_tasks.append(task)
        
        
    
    orchestra = ModelOrchestrator(all_tasks)
    orchestra.verbose = 1
    orchestra.buffer = 17000
    orchestra.generate()
    for p in profilers:
        p.start_profile()
    time = orchestra.train_models()
    for prof in profilers:
        prof.print_model_profile()
        prof.end_profile()

    lowest_score = 1000000000
    best_model = None
    best_idx = -1
    for idx, model in enumerate(all_models):
        score = run_test(model)
        if (score < lowest_score):
            best_model = model
            lowest_score = score
            best_idx = idx
    
    lr = lr_names[best_idx / 3]
    b_size = batch_sizes[best_idx % 3 ]
    print("RANDOM SEED: {} BEST LR: {} BEST B_SIZE: {}".format(seed, lr, b_size))
    torch.save(best_model, "hydra_best_model_lr_{}_bsize_{}_seed_{}".format(lr, b_size, seed))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int)
    args = parser.parse_args()
    main(args.seed)
