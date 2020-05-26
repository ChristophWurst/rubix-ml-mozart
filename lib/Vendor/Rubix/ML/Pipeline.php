<?php

namespace Test\Vendor\Rubix\ML;

use Test\Vendor\Rubix\ML\Datasets\Dataset;
use Test\Vendor\Rubix\ML\Transformers\Elastic;
use Test\Vendor\Rubix\ML\Other\Helpers\Params;
use Test\Vendor\Rubix\ML\Transformers\Stateful;
use Test\Vendor\Rubix\ML\Other\Traits\RanksSingle;
use Test\Vendor\Rubix\ML\Transformers\Transformer;
use Test\Vendor\Rubix\ML\Other\Traits\ProbaSingle;
use Test\Vendor\Rubix\ML\Other\Traits\PredictsSingle;
use Test\Vendor\Psr\Log\LoggerInterface;
use InvalidArgumentException;
use RuntimeException;

use function get_class;

/**
 * Pipeline
 *
 * Pipeline is a meta-estimator capable of transforming an input dataset by applying a
 * series of Transformer *middleware*. Under the hood, Pipeline will automatically fit the
 * training set and transform any Dataset object supplied as an argument to one of the base
 * estimator's methods before hitting the method context. With *elastic* mode enabled,
 * Pipeline will update the fitting of Elastic transformers during partial training.
 *
 * > **Note:** Since transformations are applied to dataset objects in-place (without making a
 * copy of the data), using a dataset in a program after it has been run through Pipeline may
 * have unexpected results. If you need to keep a *clean* dataset in memory you can clone
 * the dataset object before calling the method on Pipeline that consumes it.
 *
 * @category    Machine Learning
 * @package     Rubix/ML
 * @author      Andrew DalPino
 */
class Pipeline implements Online, Wrapper, Probabilistic, Ranking, Persistable, Verbose
{
    use PredictsSingle, ProbaSingle, RanksSingle;

    /**
     * A list of transformers to be applied in order.
     *
     * @var \Test\Vendor\Rubix\ML\Transformers\Transformer[]
     */
    protected $transformers = [
        //
    ];

    /**
     * An instance of a base estimator to receive the transformed data.
     *
     * @var \Test\Vendor\Rubix\ML\Estimator
     */
    protected $base;

    /**
     * Should we update the elastic transformers during partial train?
     *
     * @var bool
     */
    protected $elastic;

    /**
     * The PSR-3 logger instance.
     *
     * @var \Test\Vendor\Psr\Log\LoggerInterface|null
     */
    protected $logger;

    /**
     * Whether or not the transformer pipeline has been fitted.
     *
     * @var bool
     */
    protected $fitted;

    /**
     * @param \Test\Vendor\Rubix\ML\Transformers\Transformer[] $transformers
     * @param \Test\Vendor\Rubix\ML\Estimator $base
     * @param bool $elastic
     * @throws \InvalidArgumentException
     */
    public function __construct(array $transformers, Estimator $base, bool $elastic = true)
    {
        foreach ($transformers as $transformer) {
            if (!$transformer instanceof Transformer) {
                throw new InvalidArgumentException('Transformer must'
                    . ' implement the Transformer interface.');
            }
        }

        $this->transformers = $transformers;
        $this->base = $base;
        $this->elastic = $elastic;
    }

    /**
     * Return the estimator type.
     *
     * @return \Test\Vendor\Rubix\ML\EstimatorType
     */
    public function type() : EstimatorType
    {
        return $this->base->type();
    }

    /**
     * Return the data types that the model is compatible with.
     *
     * @return \Test\Vendor\Rubix\ML\DataType[]
     */
    public function compatibility() : array
    {
        return $this->base->compatibility();
    }

    /**
     * Return the settings of the hyper-parameters in an associative array.
     *
     * @return mixed[]
     */
    public function params() : array
    {
        return [
            'transformers' => $this->transformers,
            'estimator' => $this->base,
            'elastic' => $this->elastic,
        ];
    }

    /**
     * Sets a logger instance on the object.
     *
     * @param \Test\Vendor\Psr\Log\LoggerInterface $logger
     */
    public function setLogger(LoggerInterface $logger) : void
    {
        if ($this->base instanceof Verbose) {
            $this->base->setLogger($logger);
        }

        $this->logger = $logger;
    }

    /**
     * Return the logger or null if not set.
     *
     * @return \Test\Vendor\Psr\Log\LoggerInterface|null
     */
    public function logger() : ?LoggerInterface
    {
        return $this->logger;
    }

    /**
     * Has the learner been trained?
     *
     * @return bool
     */
    public function trained() : bool
    {
        return $this->base instanceof Learner
            ? $this->base->trained()
            : true;
    }

    /**
     * Return the base estimator instance.
     *
     * @return \Test\Vendor\Rubix\ML\Estimator
     */
    public function base() : Estimator
    {
        return $this->base;
    }

    /**
     * Run the training dataset through all transformers in order and use the
     * transformed dataset to train the estimator.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function train(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            if ($transformer instanceof Stateful) {
                $transformer->fit($dataset);

                if ($this->logger) {
                    $this->logger->info('Fitted ' . Params::shortName(get_class($transformer)));
                }
            }

            $dataset->apply($transformer);

            if ($this->logger) {
                $this->logger->info('Applied ' . Params::shortName(get_class($transformer)));
            }
        }

        if ($this->base instanceof Learner) {
            $this->base->train($dataset);
        }
    }

    /**
     * Perform a partial train.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function partial(Dataset $dataset) : void
    {
        if ($this->elastic) {
            foreach ($this->transformers as $transformer) {
                if ($transformer instanceof Elastic) {
                    $transformer->update($dataset);
    
                    if ($this->logger) {
                        $this->logger->info('Updated ' . Params::shortName(get_class($transformer)));
                    }
                }
    
                $dataset->apply($transformer);
    
                if ($this->logger) {
                    $this->logger->info('Applied ' . Params::shortName(get_class($transformer)));
                }
            }
        } else {
            $this->preprocess($dataset);
        }

        if ($this->base instanceof Online) {
            $this->base->partial($dataset);
        }
    }

    /**
     * Preprocess the dataset and return predictions from the estimator.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @return mixed[]
     */
    public function predict(Dataset $dataset) : array
    {
        $this->preprocess($dataset);

        return $this->base->predict($dataset);
    }

    /**
     * Estimate the joint probabilities for each possible outcome.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return array[]
     */
    public function proba(Dataset $dataset) : array
    {
        $this->preprocess($dataset);

        if (!$this->base instanceof Probabilistic) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Probabilistic interface.');
        }

        return $this->base->proba($dataset);
    }

    /**
     * Return the anomaly scores assigned to the samples in a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     * @throws \RuntimeException
     * @return float[]
     */
    public function rank(Dataset $dataset) : array
    {
        $this->preprocess($dataset);
        
        if (!$this->base instanceof Ranking) {
            throw new RuntimeException('Base Estimator must'
                . ' implement the Ranking interface.');
        }
            
        return $this->base->rank($dataset);
    }

    /**
     * Apply the transformer stack to a dataset.
     *
     * @param \Test\Vendor\Rubix\ML\Datasets\Dataset $dataset
     */
    public function preprocess(Dataset $dataset) : void
    {
        foreach ($this->transformers as $transformer) {
            $dataset->apply($transformer);
        }
    }

    /**
     * Allow methods to be called on the estimator from the wrapper.
     *
     * @param string $name
     * @param mixed[] $arguments
     * @return mixed
     */
    public function __call(string $name, array $arguments)
    {
        foreach ($arguments as $argument) {
            if ($argument instanceof Dataset) {
                $this->preprocess($argument);
            }
        }

        return $this->base->$name(...$arguments);
    }
}
