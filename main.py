import torch, gc
import numpy as np
import os
import time

from config import optimizer, lr
from config import model, model_idx
from config import dataset_name, training_light_field_downsample_rate, training_light_field_epoch, batch_size
from config import optimized_losses, loss_weights, estimate_clear_region, refocused_img_metrics, refocused_img_metrics_name
from config import tolerance, min_percent
from helper import EarlyStopping
from prepare_data import get_dataloaders
from model import BaselineMethod, FilterBankMethod, LinearFilter
from train import training
from test import testing

# for LinearFilter
#load_FBmodel = True

torch.set_default_dtype(torch.float32)
gc.collect()
with torch.no_grad():
    torch.cuda.empty_cache()

if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"
print(f"device: {device}")
TRAIN = True

if __name__=="__main__":
    if model == "FilterBankMethod":
        methods = [FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=model_idx)]
        #methods = [FilterBankMethod(device, 2, 2, in_channels=4, out_channels=4, kernel_size=(1, 8, 8), stride=(1, 2, 2), model_idx=model_idx)]
        methods_name = ['Filterbank']

        for params in methods[0].net.parameters():
            print(params.size())

    elif model == "LinearFilter":
        if dataset_name == "HCI":
            h, w = 512, 512
        if dataset_name == "INRIA_Lytro":
            h, w = 379, 379
        if dataset_name == "RandomTraining":
            h, w = 512, 512

        
        try:
            model = FilterBankMethod(device, 3, 3, in_channels=9, out_channels=9, kernel_size=(1, 7, 7), stride=(1, 3, 3), model_idx=model_idx)
            model.load_model(os.path.join('model',f"FilterBankMethod_{model_idx}",'best_model'))
            for params in model.net.parameters(): # only one iter for FB_kernels
                FB_kernels = params
            methods = [LinearFilter(device, h, w, s=3, t=3, model_idx=model_idx, FB_kernels=FB_kernels)]
        except:
            methods = [LinearFilter(device, h, w, s=3, t=3, model_idx=model_idx, FB_kernels=None)]
        
        
        
        
        methods_name = ['LinearFilter']


        for params in methods[0].net.parameters():
            print(params.size())

    elif model == "BaselineMethod":
        methods = [BaselineMethod(3,3)]
        methods_name = ['BaselineMethod']

        train_dataloader, test_dataloader = get_dataloaders(dataset_name, batch_size=batch_size, downsample_rate=1)

        for method_idx in range(len(methods)):
            #losses, metrics = testing(test_dataloader, methods[method_idx])
            losses, metrics = testing(test_dataloader, device, methods[method_idx])
            #plotting_dataloader(test_dataloader, methods[method_idx])
            log_str = methods[method_idx].name + ": "
        for optimized_losses_idx in range(len(optimized_losses)):
            log_str += "Loss %d: %2f " % (optimized_losses_idx, losses[optimized_losses_idx])
        for refocused_img_metrics_idx in range(len(refocused_img_metrics)):
            log_str += "Metric %s: %2f " % (refocused_img_metrics_name[refocused_img_metrics_idx], metrics[refocused_img_metrics_idx])
        
        print(log_str)
        TRAIN = False

    if TRAIN:
        training_time = []
        
        try:
            for method_idx in range(len(methods)):
                #methods[method_idx].load_model(os.path.join('model',methods[method_idx].name,'best_model'))
                # for FilterBank
                methods[method_idx].load_model(os.path.join('model','FilterBankMethod_F1','best_model'))
        except:
            pass
        
        
        
        

        downsample_rate_idx = 0
        while downsample_rate_idx < len(training_light_field_downsample_rate):
            start = time.time()
            epochs = training_light_field_epoch[downsample_rate_idx]
            downsample_rate = training_light_field_downsample_rate[downsample_rate_idx]
            try:
                #train_dataloader, test_dataloader = get_dataloaders('SRFilterTraining', batch_size=batch_size, downsample_rate=downsample_rate)
                train_dataloader, test_dataloader = get_dataloaders(dataset_name, batch_size=batch_size, downsample_rate=downsample_rate)
                
                optimizers = []
                for method_idx in range(len(methods)):
                    optimizers.append(optimizer(methods[method_idx].net.parameters(), lr = lr))
                    #methods[method_idx].clear_history()
                    methods[method_idx].train_mode()
                    if downsample_rate_idx > 0:
                        methods[method_idx].clear_history()

                early_stopped = [False for _ in range(len(methods))]
                early_stopping = EarlyStopping(tolerance=tolerance/10, min_percent=min_percent)

                for epoch in range(0,epochs+1):
                    if early_stopped == [True for _ in range(len(methods))]:
                        break
                        
                    #training(train_dataloader,device,methods,optimizers,optimized_losses,estimate_clear_region,early_stopped)

                    if epoch%10==0:
                        with torch.no_grad():
                            for method_idx in range(len(methods)):
                                if early_stopped[method_idx] == True:
                                    print(f"{methods[method_idx].name} early stopped")
                                    continue
                                else:
                                    methods[method_idx].eval_mode()
                                    losses, metrics = testing(test_dataloader, device, methods[method_idx], epoch, estimate_clear_region)
                                
                                    methods[method_idx].record.loss_history.append([])
                                    methods[method_idx].record.metric_history.append([])
                                    log_str = "Downsample %d Epoch [%d/%d] %s: " % (downsample_rate, epoch, epochs, methods[method_idx].name)
                                    for optimized_losses_idx in range(len(optimized_losses)):
                                        methods[method_idx].record.tb_writer.add_scalar("Loss/loss_sum", losses[optimized_losses_idx], epoch)
                                        methods[method_idx].record.loss_history[-1].append(losses[optimized_losses_idx])
                                        log_str += "Loss %d: %.6f " % (optimized_losses_idx, methods[method_idx].record.loss_history[-1][-1])
                                    for refocused_img_metrics_idx in range(len(refocused_img_metrics)):
                                        methods[method_idx].record.tb_writer.add_scalar("Metric/%s"%refocused_img_metrics_name[refocused_img_metrics_idx], metrics[refocused_img_metrics_idx], epoch)
                                        methods[method_idx].record.metric_history[-1].append(metrics[refocused_img_metrics_idx])
                                        log_str += "Metric %s: %.6f " % (refocused_img_metrics_name[refocused_img_metrics_idx], methods[method_idx].record.metric_history[-1][-1])

                                    if np.sum(methods[method_idx].record.loss_history[-1]) < methods[method_idx].record.best_loss:
                                        print("Found better model: %.6f < %.6f" % (np.sum(methods[method_idx].record.loss_history[-1]), methods[method_idx].record.best_loss))
                                        methods[method_idx].record.best_loss = np.sum(methods[method_idx].record.loss_history[-1])
                                        methods[method_idx].save_model(os.path.join('model',methods[method_idx].name,'best_model'))

                                    print(log_str)
                                    
                                    if epoch >= 100:
                                        test_loss = losses[0]
                                        early_stopping(test_loss)
                                        print(f"Last_loss:{early_stopping.last_loss}, Counter:{early_stopping.counter}")
                                        if early_stopping.early_stop:
                                            print("Early stopped at epoch: ", epoch)
                                            methods[method_idx].record.tb_writer.close()
                                            early_stopped[method_idx] = True
                                            break
                                    


        
                downsample_rate_idx += 1

                end = time.time()
                time_in_min = (end-start)//60
                training_time.append(int(time_in_min))

            
            except RuntimeError:
                batch_size //= 2
                print("Batchsize too large. Use batchsize = %d"% batch_size)
                gc.collect()
                with torch.no_grad():
                    torch.cuda.empty_cache()
                
                if batch_size<1:
                    print("Batchsize is zero!")
                    break
            
            

        print(training_time)
        for i in range(len(training_time)):
            print('{:02d}:Time: {:02d}:{:02d}'.format(i, training_time[i]//60, training_time[i]%60))