import torch
from utils import get_data_loader, pretraining_loss


def main(model):
  print("Loading model {}".format(model))
  model = torch.load(model)
  model.cuda()
  data_loader = get_data_loader(1)
  with torch.no_grad():
  with torch.no_grad():
        accum_loss = 0
        for sample, label in data_loader:
            ctr+=1
            outputs = model(sample)
            my_loss = pretraining_loss(outputs, label)
            accum_loss += my_loss.item()
            if (ctr % 10 == 0):
                print("SAMPLE: {} / {}".format(ctr, len(valid_loader)))
                print("RUNNING PPL: {}".format(math.exp(accum_loss/ctr)))
    print("TEST LOSS: {}".format(math.exp(accum_loss/ctr)))
    return math.exp(accum_loss/ctr)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    main(args.model)
