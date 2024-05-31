from load_data import *
if __name__ == '__main__':
    batch = 8
    train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,fulltest_dataset = get_data(batch)
    print_data(train_dataset,train_dataloader,validation_dataset,test_dataset,validation_dataloader,test_dataloader,fulltest_dataset)