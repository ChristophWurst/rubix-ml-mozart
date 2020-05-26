<?php

namespace Test\Vendor\Rubix\ML\Transformers;

use Test\Vendor\Rubix\ML\Datasets\Dataset;

interface Elastic extends Stateful
{
    /**
     * Update the fitting of the transformer.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function update(Dataset $dataset) : void;
}
