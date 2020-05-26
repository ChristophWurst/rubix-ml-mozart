<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer;

interface Output extends Layer
{
    /**
     * Compute the gradient and loss at the output.
     *
     * @param (string|int|float)[] $labels
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Optimizers\Optimizer $optimizer
     * @throws \RuntimeException
     * @return mixed[]
     */
    public function back(array $labels, Optimizer $optimizer) : array;
}
