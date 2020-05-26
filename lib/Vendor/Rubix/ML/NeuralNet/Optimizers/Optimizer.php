<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Optimizers;

use TEST_Tensor\Tensor;
use Test\Vendor\Rubix\ML\NeuralNet\Parameter;

interface Optimizer
{
    /**
     * Take a step of gradient descent for a given parameter.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter $param
     * @param \TEST_Tensor\Tensor<int|float|array> $gradient
     * @return \TEST_Tensor\Tensor<int|float|array>
     */
    public function step(Parameter $param, TEST_Tensor $gradient) : TEST_Tensor;
}
