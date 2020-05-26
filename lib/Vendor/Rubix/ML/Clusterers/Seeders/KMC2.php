<?php

namespace Test\Vendor\Rubix\ML\Clusterers\Seeders;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Kernels\Distance\Distance;
use Test\Vendor\Rubix\ML\Kernels\Distance\Euclidean;
use InvalidArgumentException;

use function count;

use const Test\Vendor\Rubix\ML\EPSILON;

/**
 * K-MC2
 *
 * A fast Plus Plus approximator that replaces the brute force method with a substantially
 * faster Markov Chain Monte Carlo (MCMC) sampling procedure with comparable results.
 *
 * References:
 * [1] O. Bachem et al. (2016). Approximate K-Means++ in Sublinear Time.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class KMC2 implements Seeder
{
    /**
     * The number of candidate nodes in the Markov Chain.
     *
     * @var int
     */
    protected $m;

    /**
     * The distance kernel used to compute the distance between samples.
     *
     * @var \Test\Vendor\Rubix\ML\Kernels\Distance\Distance
     */
    protected $kernel;

    /**
     * @param int $m
     * @param \Test\Vendor\Rubix\ML\Kernels\Distance\Distance|null $kernel
     * @throws \InvalidArgumentException
     */
    public function __construct(int $m = 50, ?Distance $kernel = null)
    {
        if ($m < 1) {
            throw new InvalidArgumentException('M must be greater'
                . " than 0, $m given.");
        }

        $this->m = $m;
        $this->kernel = $kernel ?? new Euclidean();
    }

    /**
     * Seed k cluster centroids from a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @param int $k
     * @throws \RuntimeException
     * @return array[]
     */
    public function seed(Dataset $dataset, int $k) : array
    {
        $max = getrandmax();

        $centroids = $dataset->randomSubsetWithReplacement(1)->samples();

        while (count($centroids) < $k) {
            $target = end($centroids) ?: [];
            
            $candidates = $dataset->randomSubsetWithReplacement($this->m)->samples();

            $x = array_pop($candidates) ?? [];

            $xDistance = $this->kernel->compute($x, $target) ?: EPSILON;

            foreach ($candidates as $candidate) {
                $yDistance = $this->kernel->compute($candidate, $target);

                $density = min(1.0, $yDistance / $xDistance);

                $threshold = rand() / $max;

                if ($density > $threshold) {
                    $xDistance = $yDistance;

                    $x = $candidate;
                }
            }

            $centroids[] = $x;
        }

        return $centroids;
    }
}
