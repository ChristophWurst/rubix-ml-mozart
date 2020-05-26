<?php

namespace Test\Vendor\Rubix\ML\NeuralNet\Layers;

use Generator;

interface Parametric
{
    /**
     * Return the parameters of the layer.
     *
     * @return \Generator<\Test\Vendor\Rubix\ML\NeuralNet\Parameter>
     */
    public function parameters() : Generator;

    /**
     * Restore the parameters on the layer from an associative array.
     *
     * @param \Test\Vendor\Rubix\ML\NeuralNet\Parameter[] $parameters
     */
    public function restore(array $parameters) : void;
}
