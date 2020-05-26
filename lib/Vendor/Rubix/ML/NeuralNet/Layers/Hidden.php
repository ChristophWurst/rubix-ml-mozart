<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use Test\Vendor\Rubix\ML\Deferred;
use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Hidden extends Layer
{
    /**
     * Calculate the gradient and update the parameters of the layer.
     *
     * @param \Test\Vendor\Rubix\ML\Deferred $prevGradient
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @return \Test\Vendor\Rubix\ML\Deferred
     */
    public function back(Deferred $prevGradient, Optimizer $optimizer) : Deferred;
}
