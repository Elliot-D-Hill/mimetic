API reference
=============

Covariance
----------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.AR1Covariance
   mimetic.IsotropicCovariance
   mimetic.LKJCovariance
   mimetic.random_effects_covariance
   mimetic.residual_covariance

States
------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.PredictorState
   mimetic.EventTimeState
   mimetic.CensoredState
   mimetic.ObservedState
   mimetic.SurvivalState
   mimetic.TokenizedState
   mimetic.EventProcessState
   mimetic.CompetingRisksState
   mimetic.RiskIndicatorState
   mimetic.DiscreteRiskState

Predictor
---------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.linear_predictor
   mimetic.random_effects
   mimetic.activation
   mimetic.linear
   mimetic.mlp
   mimetic.observation_time

Response distributions
----------------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.gaussian
   mimetic.poisson
   mimetic.bernoulli
   mimetic.categorical
   mimetic.ordinal
   mimetic.tokens

Survival
--------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.event_time
   mimetic.censor_time
   mimetic.mixture_cure_censoring
   mimetic.survival_indicators

Competing risks
---------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.competing_risks
   mimetic.independent_events
   mimetic.risk_indicators
   mimetic.multi_event
   mimetic.discretize_risk

Simulation steps
----------------

.. autosummary::
   :toctree: api/generated
   :nosignatures:

   mimetic.Simulation
   mimetic.ResponseStep
   mimetic.DiscreteResponseStep
   mimetic.TokenizedStep
   mimetic.EventTimeStep
   mimetic.CensoredStep
   mimetic.SurvivalStep
   mimetic.CompetingRisksStep
   mimetic.IndependentEventsStep
   mimetic.RiskIndicatorStep
   mimetic.DiscreteRiskStep
   mimetic.DiscreteEventTimeStep
