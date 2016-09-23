# LogisticRegressionSandbox
A quick ML algorithm implementation on randomly-generated data

##Synopsis
This code generates 100 random data points based on a randomly generated target function, to update the weights. It then calculates an in sample error using the Cross-Entropy error measurement. Then creates 100 new data sample based on the original target function to approximate the out of sample error.

###Some Notes
Please note: This is just me sandboxing a bit, and there are a good number of inefficiencies in the code. If I get around to it, I'll fix them, but seeing as this was just some sandboxing, there's not a whole lot of motivation to do so, especially considering the work that would be required to fix some of the errors.

First, I realize that class-based inputs rather than having a proper input array of size NxD was not the smartest thing on earth. I was thinking too much like a programmer (The sort of "a class for everything and everything in its class" mentality) and less in terms of machine learning. The code works fine, but it's a little ugly in terms of calculations and not too scaleable as a result. I figured that because the project was so small, it wouldn't matter too much, and, in that sense, I was correct. Just don't try and fork this only to realize later that it doesn't scale too well.

Also, I realize real data would have been an ideal to test on rather than randomly generated points as there is not enough noise introduced to properly test the algorithm. I realize this, but the point of this exercise was for my own benefit, not to win the Netflix competition. 

The code works, it does what it's supposed to, just realize this was done more as an exercise than anything else. With that in mind, take it for what it is, and enjoy.
