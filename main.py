import argparse
from training import Solver


def main(FLAGS):
    net = Solver(FLAGS)
    
    if FLAGS.train:
        net.Train()
    else:
        return net.EvaluateScore()
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--cuda', type=bool, default=True, help="run the following code on a GPU")
    parser.add_argument('--gpu', type=int, default=0, help="id of the GPU to utilize")
    
    parser.add_argument('--train', dest='train', action='store_true', help="Trains the network")
    parser.add_argument('--eval', dest='train', action='store_false', help="Evaluates the network over the whole trainset")
    parser.set_defaults(train = True)
    
    parser.add_argument('--batch_size', type=int, default=10, help="batch size for training")
    parser.add_argument('--data_path', type=str, default='data/', help="Path to image data")
    
    parser.add_argument('--learning_rate', type=float, default=0.001, help="learning rate")
    parser.add_argument('--beta_1', type=float, default=0.9, help="default beta_1 val for adam")
    parser.add_argument('--beta_2', type=float, default=0.999, help="default beta_2 val for adam")
    
    parser.add_argument('--load_path', type=str, default='trained_model', help="model path within the checkpoint folder")
    
    parser.add_argument('--end_epoch', type=int, default=400, help="flag to indicate the final epoch of training")
        
        
    FLAGS = parser.parse_args()
    all_scores = main(FLAGS)