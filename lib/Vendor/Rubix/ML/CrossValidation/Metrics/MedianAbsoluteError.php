<?php

namespace Test\Vendor\Rubix\ML\CrossValidation\Metrics;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\EstimatorType;
use Test\Vendor\Rubix\ML\Other\Helpers\Stats;
use InvalidArgumentException;

use function count;

/**
 * Median Absolute Error
 *
 * Median Absolute Error (MAD) is a robust measure of error, similar to MAE, that ignores
 * highly erroneous predictions. Since MAD is a robust statistic, it works well even when
 * used to measure non-normal distributions.
 *
 * > **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MedianAbsoluteError implements Metric
{
    /**
     * Return a tuple of the min and max output value for this metric.
     *
     * @return float[]
     */
    public function range() : array
    {
        return [-INF, 0.0];
    }

    /**
     * The estimator types that this metric is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType[]
     */
    public function compatibility() : array
    {
        return [
            EstimatorType::regressor(),
        ];
    }

    /**
     * Score a set of predictions.
     *
     * @param (int|float)[] $predictions
     * @param (int|float)[] $labels
     * @throws \InvalidArgumentException
     * @return float
     */
    public function score(array $predictions, array $labels) : float
    {
        if (count($predictions) !== count($labels)) {
            throw new InvalidArgumentException('Number of predictions'
                . ' and labels must be equal.');
        }

        if (empty($predictions)) {
            return 0.0;
        }

        $errors = [];

        foreach ($predictions as $i => $prediction) {
            $errors[] = abs($labels[$i] - $prediction);
        }

        return -Stats::median($errors);
    }
}
