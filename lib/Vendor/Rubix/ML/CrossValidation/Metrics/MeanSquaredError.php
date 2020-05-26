<?php

namespace Test\Vendor\Rubix\ML\CrossValidation\Metrics;

use Test\Vendor\Rubix\ML\Estimator;
use Test\Vendor\Rubix\ML\EstimatorType;
use InvalidArgumentException;

use function count;

/**
 * Mean Squared Error
 *
 * A scale-dependent regression metric that gives greater weight to error scores the worse
 * they are. Formally, Mean Squared Error (MSE) is the average of the squared differences
 * between a set of predictions and their target labels.
 *
 * > **Note:** In order to maintain the convention of *maximizing* validation scores,
 * this metric outputs the negative of the original score.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class MeanSquaredError implements Metric
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

        $error = 0.0;

        foreach ($predictions as $i => $prediction) {
            $error += ($labels[$i] - $prediction) ** 2;
        }

        return -($error / count($predictions));
    }
}
