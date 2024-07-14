import torch
from torch.utils.data import DataLoader
from data import ImageTextDataset
from model import VisionTextModel
from train import train_model, evaluate_model
from test import test_model
import argparse
from utils import plot_training_history

def parse_args():
    parser = argparse.ArgumentParser(description='Train VisionTextModel')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--mode', type=str, choices=['80', '50', '20', 'test'], required=True, help='Mode: train split ratio or test')
    parser.add_argument('--resume_training', action='store_true', help='Flag to resume training from checkpoint')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')

    #cross-modal attention
    parser.add_argument('--text_hidden_dim', type=int, default=512, help='Hidden size of the text(MLP output)')
    parser.add_argument('--cross_attention_hidden_size', type=int, default=512, help='Hidden size of the cross_attention')
    parser.add_argument('--nlevels', type=int, default=2, help='Number of levels to repeat the attention mechanism')
    parser.add_argument('--attn_dropout', type=float, default=0.05, help='Attention dropout')
    parser.add_argument('--relu_dropout', type=float, default=0.1, help='ReLU dropout')
    parser.add_argument('--res_dropout', type=float, default=0.1, help='Residual block dropout')
    parser.add_argument('--out_dropout', type=float, default=0.0, help='Output layer dropout')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of heads for the transformer network (default: 3)')

    #training
    parser.add_argument('--batch_size', type=int, default=64, metavar='N', help='Batch size (default: 512)')
    parser.add_argument('--clip', type=float, default=0.8, help='Gradient clip value (default: 0.8)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam', help='Optimizer to use (default: Adam)')
    parser.add_argument('--num_epochs', type=int, default=40, help='Number of epochs (default: 40)')
    parser.add_argument('--when', type=int, default=20, help='When to decay learning rate (default: 20)')
    
    #resume model,history
    parser.add_argument('--model_path', type=str, default='../model/best_model.pth', help='Path to resume the best model')
    parser.add_argument('--history_path', type=str, default='../history/training_history.json', help='Path to resume the training history')
    parser.add_argument('--plot_path', type=str, default='training_plot.png', help='Path to save the training plot')


    '''
    #not used
    parser.add_argument('--activation_function', type=str, default='relu', choices=['relu', 'gelu', 'tanh'], help='Activation function to use')    
    parser.add_argument('--layer_norm', type=bool, default=True, help='Whether to use layer normalization')
    parser.add_argument('--lr_decay', type=float, default=0.1, help='Learning rate decay factor')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay factor')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='Number of steps to accumulate gradients')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps for learning rate scheduler')
    parser.add_argument('--intermediate_size', type=int, default=2048, help='Intermediate size of the MLP layer')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of transformer layers')
    '''

    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.mode.isdigit():
        # Load data
        datasets = ImageTextDataset(data_path=args.data_path, split=args.mode)
        train_dataset, val_dataset = datasets.train, datasets.val
        num_classes = train_dataset.get_num_classes()
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

        # Initialize model
        model = VisionTextModel(
            num_classes=num_classes,
            text_output_dim=args.text_hidden_dim, 
            cross_attention_hidden_size=args.cross_attention_hidden_size,
            nlevels=args.nlevels,
            attn_dropout=args.attn_dropout,
            relu_dropout=args.relu_dropout,
            res_dropout=args.res_dropout,
            out_dropout=args.out_dropout,
            num_heads=args.num_heads
        )

        # Training
        if args.resume_training:
            model_name = args.model_path
            history_name = args.history_path
        else:
            model_name = '../model/'+args.data_path.split('/')[-1]+args.mode+'.pth'
            history_name = '../history/'+args.data_path.split('/')[-1]+args.mode+'.json'
        best_model_path, history_path = train_model(
            model, train_loader, val_loader,
            num_epochs=args.num_epochs,
            lr=args.lr,
            optim_name=args.optim,
            clip=args.clip,
            when=args.when,
            model_path=model_name, 
            history_path=history_name,
            resume_training=args.resume_training,
            patience = args.patience,  
        )

        print(f'Training complete. Best model saved at: {best_model_path}, traning history saved at: {history_path}.')

        # Evaluation
        model.load_state_dict(torch.load(best_model_path))
        evaluate_model(model, val_loader)

        plot_training_history(history_path, output_path=args.plot_path)
        print(f'Training plot saved at: {args.plot_path}')
    
    elif args.mode == 'test':
        test_dataset = ImageTextDataset(data_path=args.data_path, split='test')
        num_classes = test_dataset.get_num_classes()
        classes = test_dataset.get_class_names()
        
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Initialize model
        model = VisionTextModel(
            num_classes=num_classes,
            text_output_dim=args.text_hidden_dim, 
            cross_attention_hidden_size=args.cross_attention_hidden_size,
            nlevels=args.nlevels,
            attn_dropout=args.attn_dropout,
            relu_dropout=args.relu_dropout,
            res_dropout=args.res_dropout,
            out_dropout=args.out_dropout,
            num_heads=args.num_heads
        )
        
        # Load the best model for evaluation
        model.load_state_dict(torch.load(args.model_path))
        
        # Evaluate on test set
        overall_acc, avg_acc, kappa = test_model(
            model, test_loader, classes,
        )
        
        print(f'Test Overall Accuracy: {overall_acc}')
        print(f'Test Average Accuracy: {avg_acc}')
        print(f'Test Kappa Coefficient: {kappa}')

if __name__ == '__main__':
    main()
