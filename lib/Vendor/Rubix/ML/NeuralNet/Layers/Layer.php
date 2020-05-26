<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use TEST_Tensor\TEST_Matrix;

interface Layer
{
    /**
     * The width of the layer. i.e. the number of neurons or computation nodes.
     *
     * @return int
     */
    public function width() : int;

    /**
     * Initialize the layer with the fan in from the previous layer and return
     * the fan out for this layer.
     *
     * @param int $fanIn
     * @return int
     */
    public function initialize(int $fanIn) : int;

    /**
     * Feed the input forward to the next layer in the network.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function forward(TEST_Matrix $input) : TEST_Matrix;

    /**
     * Forward pass during inference.
     *
     * @param \TEST_Tensor\TEST_Matrix $input
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function infer(TEST_Matrix $input) : TEST_Matrix;
}
