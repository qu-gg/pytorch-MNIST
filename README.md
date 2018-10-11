# PyTorch MNIST Implementations
<h3>Implementation of varied MNIST architectures to test PyTorch</h3>
<hr>
<ul>
<li>fc_model.py: A simple 3-layer network used to test the basics of loading the MNIST dataset and executing a training loop</li>
<li>conv_model.py: Similar in structure to the FC model, just with the addition of convolutional layers and the ability to use the network for custom predictions.</li>
<li>gan_model.py: A take on creating a Generative Adversarial Network, taking inspiration from the DCGAN model and Brandon Amos's <a href="https://bamos.github.io/2016/08/09/deep-completion/#step-1-interpreting-images-as-samples-from-a-probability-distribution">"Image Completion with Deep Learning in Tensorflow."</a></li>
</ul>

<p></p>
<h4>Optimizations for the GAN Model</h4>

<h5>Kernel Sizes</h5>
<p>One of the flaws that degraded the accuracy and quality of the images produced was the kernel sizes in both the discriminator and generator being too large. Instead of harnessing a deeper network, the kernel sizes were tweaked via trial-and-error in order to produce the desired image shape.</p>

<p>I believe this resulted in a lack of fine edges and detail in the generator due to the kernels providing too much information per convolution, distorting the distinct boundaries betweeen digit and whitespace.</p>

<p>Results from training on MNIST 8's and 9's:<p>
<img src="https://raw.githubusercontent.com/qu-gg/pytorch-MNIST/master/results/8/77epoch0num.jpg"></img></br>
<img src="https://raw.githubusercontent.com/qu-gg/pytorch-MNIST/master/results/9/66epoch17num.jpg"></img>


<h5>Optimizer, LR, Loss</h5>
<p>Some other potential optimizations is to adjust the training cycle and learning rates of the model. Because I was running on a slower set-up, I was reluctant to try a lower learning rates and larger batch sizes - though I feel like tuning these could provide for better results.</p>
 
<h4> Take-aways </h4>
<p>One of the biggest takeaways from this was that in the context of using transpose convolutions to generate images of a specified size, it is far easier to start with a randomized latent vector that that's square root is multiple of the size you are trying to reach. Using large and varying kernel sizes has the potential to mix distinct spatial information, leading to blurry results.</p>

