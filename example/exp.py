from time import time
from mxnet import autograd, nd
import mxnet as mx


from mxnet import profiler
profiler.set_config(profile_all=True, aggregate_stats=True, filename='profile_output.json')

from mxnet import gluon
net = gluon.nn.HybridSequential()
with net.name_scope():
    net.add(gluon.nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
    net.add(gluon.nn.Flatten())
    net.add(gluon.nn.Dense(512, activation="relu"))
    net.add(gluon.nn.Dense(10))



from mxnet.gluon.data.vision import transforms
train_data = gluon.data.DataLoader(gluon.data.vision.MNIST(train=True).transform_first(transforms.ToTensor()),
                                   batch_size=64, shuffle=True)



# Use GPU if available
if len(mx.test_utils.list_gpus())!=0:
    ctx=mx.gpu()
else:
    ctx=mx.cpu()

# Initialize the parameters with random weights
net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

# Use SGD optimizer
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})

# Softmax Cross Entropy is a frequently used loss function for multi-classs classification
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

# A helper function to run one training iteration
def run_training_iteration(data, label):
    
    # Load data and label is the right context
    data = data.as_in_context(ctx)
    label = label.as_in_context(ctx)
    
    # Run the forward pass
    with autograd.record():
        output = net(data)
        loss = softmax_cross_entropy(output, label)
    
    # Run the backward pass
    loss.backward()
    
    # Apply changes to parameters
    trainer.step(data.shape[0])



# Run the first iteration without profiling
itr = iter(train_data)
run_training_iteration(*next(itr))




data, label = next(itr)

# Ask the profiler to start recording
profiler.set_state('run')

run_training_iteration(*next(itr))

# Ask the profiler to stop recording after operations have completed
mx.nd.waitall()
profiler.set_state('stop')

# print('aaaaa')


# print(profiler.getSummary())
# print(profiler.getSummary())
# profiler.clear()


# profiler.set_state('run')

# run_training_iteration(*next(itr))

# # Ask the profiler to stop recording after operations have completed
# mx.nd.waitall()
# profiler.set_state('stop')




# print(profiler.dumps())
# print('aaa')
# print(profiler.dumps())
# profiler.dump(profile_process = 'server')

# profiler.set_state('run')

# run_training_iteration(*next(itr))

# # Ask the profiler to stop recording after operations have completed
# mx.nd.waitall()
# profiler.set_state('stop')

print(profiler.dumps())
# print(profiler.get_summary())
# profiler.reset()
# print(profiler.getSummary())

# profiler.set_state('run')
# run_training_iteration(*next(itr))
# # Ask the profiler to stop recording after operations have completed
# mx.nd.waitall()
# profiler.set_state('stop')


# print(profiler.getSummary())
# profiler.dump()



# # Ask the profiler to start recording
# profiler.set_state('run')

# run_training_iteration(*next(itr))

# # Ask the profiler to stop recording after operations have completed
# mx.nd.waitall()
# profiler.set_state('stop')

# print(profiler.dumps())
# print(profiler.dumps())

# profiler.dump(finished=True)
# profiler.dump(finished=False)
# print(profiler.getSummary())