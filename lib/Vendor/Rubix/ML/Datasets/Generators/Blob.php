<?php

namespace Test\Vendor\Rubix\ML\Datasets\Generators;

use TEST_Tensor\TEST_Matrix;
use TEST_Tensor\TEST_Vector;
use Test\Vendor\Rubix\ML\Datasets\Unlabeled;
use InvalidArgumentException;

use function count;

/**
 * Blob
 *
 * A normally distributed n-dimensional blob of samples centered at a given
 * mean vector. The standard deviation can be set for the whole blob or for each
 * feature column independently. When a global standard deviation is used, the
 * resulting blob will be isotropic and will converge asymptotically to a sphere.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Blob implements Generator
{
    /**
     * The center vector of the blob.
     *
     * @var \TEST_Tensor\TEST_Vector
     */
    protected $center;

    /**
     * The standard deviation of the blob.
     *
     * @var \TEST_Tensor\TEST_Vector|int|float
     */
    protected $stddev;

    /**
     * @param (int|float)[] $center
     * @param int|float|(int|float)[] $stddev
     * @throws \InvalidArgumentException
     */
    public function __construct(array $center = [0, 0], $stddev = 1.0)
    {
        if (empty($center)) {
            throw new InvalidArgumentException('Cannot generate samples'
                . ' with dimensionality less than 1.');
        }

        if (is_array($stddev)) {
            if (count($center) !== count($stddev)) {
                throw new InvalidArgumentException('Number of center'
                    . ' coordinates and standard deviations must be equal.');
            }

            foreach ($stddev as $value) {
                if ($value < 0) {
                    throw new InvalidArgumentException('Standard deviation'
                        . " must be greater than 0, $value given.");
                }
            }

            $stddev = TEST_Vector::quick($stddev);
        } else {
            if ($stddev <= 0) {
                throw new InvalidArgumentException('Standard deviation'
                    . " must be greater than 0, $stddev given.");
            }
        }

        $this->center = TEST_Vector::quick($center);
        $this->stddev = $stddev;
    }

    /**
     * Return the dimensionality of the data this generates.
     *
     * @return int
     */
    public function dimensions() : int
    {
        return $this->center->n();
    }

    /**
     * Generate n data points.
     *
     * @param int $n
     * @return \Test\Vendor\Rubix\ML\Datasets\Unlabeled
     */
    public function generate(int $n) : Unlabeled
    {
        $d = $this->dimensions();
        
        $samples = TEST_Matrix::gaussian($n, $d)
            ->multiply($this->stddev)
            ->add($this->center)
            ->asArray();

        return Unlabeled::quick($samples);
    }
}
