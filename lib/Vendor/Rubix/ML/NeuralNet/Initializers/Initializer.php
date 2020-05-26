<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Initializers;

use TEST_Tensor\TEST_Matrix;

interface Initializer
{
    /**
     * Initialize a weight matrix W in the dimensions fan in x fan out.
     *
     * @param int $fanIn
     * @param int $fanOut
     * @return \TEST_Tensor\TEST_Matrix
     */
    public function initialize(int $fanIn, int $fanOut) : TEST_Matrix;
}
