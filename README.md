# ece272b-homework-2-deepdream-solved
**TO GET THIS SOLUTION VISIT:** [ECE272B Homework 2-DeepDream Solved](https://www.ankitcodinghub.com/product/ece272b-homework-2-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;100178&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;0&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;0&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;0\/5 - (0 votes)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;ECE272B Homework 2-DeepDream Solved&quot;,&quot;width&quot;:&quot;0&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 0px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            <span class="kksr-muted">Rate this product</span>
    </div>
    </div>
<div class="page" title="Page 1">
<div class="layoutArea">
<div class="column">
Introduction

In the last several lectures, we have seen how convolutional neural networks (CNN) can be applied to various image-based machine learning tasks. However, as with most neural networks, CNN’s can act like a black box. We know WHAT we train it on (dataset, X, y) , we know HOW it trains (i.e. gradient descent), and we can observe the performance (cross validation). What is difficult to grasp though, is what the (C)NN actually learned.

For this homework we will implement “DeepDream”, described in this blog post, both as a method to visualize the patterns learned by a network and to create interesting images. Here is the corresponding Tensorflow tutorial. You will also utilize free GPU/TPU resources available on Google Cloud to speed up computations.

Google CoLab

Google Colab is a Python development environment that runs in the browser using Google Cloud. It is similar Jupyter Notebook but with the latest Tensorflow package pre-installed. You can learn about CoLab’s basic features here. We will be mainly touching three features of Colab in this homework:

• Online python editing environment based on Jupyter.

• Virtual machine with private file system that’s associated with your Google account. • Free GPU and TPU resources.

Data Set

The dataset can be any images you want! The provided .ipynb contains URLs as starting images, which you may use. Here’s the code snippet for loading an image from a URL and clipping it to an smaller size:

<pre>    # Download an image and read it into a NumPy array.
</pre>
def download(url, max_dim=None):

name = url.split(‘/’)[-1]

image_path = tf.keras.utils.get_file(name, origin=url)

img = PIL.Image.open(image_path)

# if a max dimension is given, resize the image to smaller size. if max_dim:

<pre>            img.thumbnail((max_dim, max_dim))
        return np.array(img)
</pre>
The data you work with will also include a purely noise image, which you will have to generate. Remember, RGB has 3 layers and takes integer values from 0 to 255.

<pre>    # Return a noisy RGB image
</pre>
def noisy_image(shape=(500,500,3)): # TODO: noise = np.random. … # TODO: scaling, datatype, etc return noise

</div>
</div>
<div class="layoutArea">
<div class="column">
1

</div>
</div>
</div>
<div class="page" title="Page 2">
<div class="layoutArea">
<div class="column">
Workflow

</div>
</div>
<div class="layoutArea">
<div class="column">
Figure 1: Example set of original Images

</div>
</div>
<div class="layoutArea">
<div class="column">
You must perform the following guiding tasks and write down answers to the following questions:

All students (157B/272B)

1. (5 pts) Prepare data and model

<ul>
<li>(1 pt) Select at least 2 images, and download them via URL. Cut down to a maximum size and display them.</li>
<li>(1 pt) Generate a RGB image of random noise, of comparable size.</li>
<li>(1 pt) Download InceptionV3 model for ImageNet.</li>
<li>(2 pts) Write a deprocess function to convert images from the format of InceptionV3 (float -1.0-1.0) back to RGB image format (int 0-255).2. (5 pts) Select “dream” layers</li>
</ul>
<ul>
<li>(2 pts) Extract 1-4 layers of interest from InceptionV3, and create a Tensorflow model that takes image input and outputs the activations of these layers.</li>
<li>(1 pts) Record which layers you (initially) picked to “dream” from.</li>
<li>(2 pts) Explain what you expect to happen to the output image as you pick layers that are deeper (layer 0 is the shallowest, layer 11 is the deepest). (hint: think back to CNN’s in HW 1 – what layer usually follows a convolution, and what does it do?)3. (8 pts) Develop the loss function</li>
</ul>
<ul>
<li>(1 pts) What is the loss function in DeepDream?</li>
<li>(2 pts) What normalizing step(s) is taken in the loss function? Why? (hint: activations are the output of layers – what can differ if you randomly pick two layers to compare?)</li>
<li>(2 pts) Write a function to calculate the loss from passing an image through the model.</li>
<li>(3 pts) Are we trying to maximize or minimize this loss? Why does your answer make sense in terms of visualizing what the network has learned? (hint: read the blog post)4. (6 pts) Complete the DeepDream algorithm</li>
</ul>
• (4 pts) In homework 1, our loss relates to the error: L ∝ E = prediction − true. As a

result the gradient was computed as dL , relating the error (E) and the network weights dw

(w). By adjusting the weights to descend this gradient, we hoped for the loss and therefore the error to decrease.

<ul>
<li>– &nbsp;(2 pts) What gradient are we computing in the DeepDream algorithm? (i.e. is it still gradient of error with respect to weights, or something else)?</li>
<li>– &nbsp;(2 pts) Are we still using gradient descent? Explain how the dream algorithm works. (hint: the gradient relates two values – we are trying to maximize/minimize one of those values, so how do we use the gradient?).</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
2

</div>
</div>
</div>
<div class="page" title="Page 3">
<div class="layoutArea">
<div class="column">
• (2 pts) Within the DeepDream(tf.Module) in the notebook, complete the algorithm by appropriately using the calculating the gradient and using as you answered above.

<ol start="5">
<li>(4 pts) Write run_deep_dream_simple as a “main” function that takes ONE image: prepro- cess as necessary, and run the DeepDream algorithm on it. Allow for a variable number of iterations. Have the function output the images and losses at regular intervals.</li>
<li>(4 pts) Run your algorithm on each image. (Hint: make sure runtime –&gt; change runtime type –&gt; GPU to speed up the processing. CPU only will take about 1 minute per 100 steps per image). Display the evolution of your images – do at least 150 steps and display the progress at every 50. Caption them with #steps and loss. The show_side_by_side function you (optionally) write at the top can take a list of images and may be helpful.</li>
<li>(3 pts) Read the blog post, if you haven’t yet. Now that you’ve seen your results for an image of all noise, describe the differences with the noisy image dream in the blog post. Can you explain why these differences are so distinct? (hint – remember the gradients. What do our gradients represent? ImageNet is a dataset for image classification, so what gradients might the blog post be using instead to show such results?)</li>
<li>(3 pts) Rerun your algorithm on at least one image with different parameters (layer(s), step_size, skip gradient normalization, etc…) to try generate something noticeably different. Display results and make any comments. If you cannot find significantly different examples then “Surprisingly, nothing changed” is perfectly valid – BUT include thoughts on why.BONUS: If you can generate especially interesting images from your noise, you will receive some bonus points. However, this will require a lot of work and should be pursued only if the topic is very interesting to you. Start with the octaves step detailed below. Examples of images generated purely from noise from the blog post are shown in the following figure, as a (exemplary) benchmark for “interesting”.
Figure 2: Images dreamed from noise. Source
</li>
</ol>
Grad (272B)

1. (6pts) Add “octaves” to the Algorithm

<ul>
<li>(3 pts) Read the section from the Tensorflow tutorial on octaves. Why does resizing the image as we go “allow patterns generated at smaller scales to be incorporated into patterns at higher scales and filled in with additional detail.”. (hint: we are still looking at the same layers as in the basic algorithm, so depth is not a factor.)</li>
<li>(3 pts) Write run_deep_dream_octaves as a “main” function that takes ONE image and performs the DeepDream with “octaves” algorithm. Allow for a variable number of octaves and variable steps per octave. Have the function output images and losses for each octave. (You should be able to easily incorporate run_deep_dream_simple within this function).</li>
</ul>
</div>
</div>
<div class="layoutArea">
<div class="column">
3

</div>
</div>
</div>
<div class="page" title="Page 4">
<div class="layoutArea">
<div class="column">
Figure 3: Dream output (including octaves)

2. (4 pts) Display the evolution of your images across the octaves. This means you will have to make sure the images are restored to their proper size after each octave. Do at least 3 octaves, caption them with octave # and loss.

</div>
</div>
<div class="layoutArea">
<div class="column">
4

</div>
</div>
</div>
