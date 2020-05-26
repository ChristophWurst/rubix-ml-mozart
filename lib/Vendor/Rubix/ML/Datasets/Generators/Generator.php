<?php

namespace Test\Vendor\Rubix\ML\Datasets\Generators;

interface Generator
{
    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int;

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Test\Vendor\Rubix\ML\Datasets\Dataset
     */
    public function generate(int $n);
}
