<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Optimizers;

use Test\Vendor\Rubix\ML\NeuralNet\Parameter;

interface Adaptive extends Optimizer
{
    /**
     * Warm the parameter cache.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter $param
     */
    public function warm(Parameter $param) : void;
}
