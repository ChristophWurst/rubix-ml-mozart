<?php

namespace Test\Vendor\Rubix\ML\Clusterers\Seeders;

use Test\Vendor\Rubix\ML\Datasets\Dataset;

interface Seeder
{
    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @return array[]
     */
    public function seed(Dataset $dataset, int $k) : array;
}
