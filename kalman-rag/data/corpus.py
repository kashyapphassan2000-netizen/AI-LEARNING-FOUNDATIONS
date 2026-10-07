"""A small, original control-theory corpus written from scratch for this demo.

Real kalman-rag ingests public-domain textbooks and NASA technical reports.
To keep this guide runnable offline and free of copyright issues, the corpus
below is original explanatory text. It is enough to make retrieval meaningful.
Each entry is one short "document" with a source label and page-like id.
"""

CORPUS = [
    {
        "source": "notes/kalman_basics",
        "page": 1,
        "text": (
            "The Kalman filter is a recursive estimator for linear dynamical systems "
            "corrupted by Gaussian noise. It maintains a state estimate and a covariance "
            "that quantifies the uncertainty of that estimate. Each cycle has two steps: "
            "a predict step that propagates the state through the system dynamics, and an "
            "update step that corrects the prediction using a new measurement."
        ),
    },
    {
        "source": "notes/kalman_gain",
        "page": 2,
        "text": (
            "The Kalman gain determines how much the measurement corrects the prediction. "
            "It is derived by minimizing the trace of the posterior error covariance. The "
            "gain equals the predicted covariance times the measurement matrix transpose, "
            "divided by the innovation covariance. A large gain trusts the measurement; a "
            "small gain trusts the model prediction. Deriving the Kalman gain is the core "
            "of the filter."
        ),
    },
    {
        "source": "notes/state_space",
        "page": 3,
        "text": (
            "A discrete-time state-space model writes the next state as the state "
            "transition matrix A times the current state plus the input matrix B times the "
            "control input, plus process noise. The output equation writes each "
            "measurement as the measurement matrix H times the state plus measurement "
            "noise. The Kalman filter's predict step uses A and B, and its update step "
            "uses H."
        ),
    },
    {
        "source": "notes/noise_covariance",
        "page": 4,
        "text": (
            "Two covariance matrices tune a Kalman filter. Q, the process noise "
            "covariance, says how much the true dynamics wander away from the model. R, "
            "the measurement noise covariance, says how noisy the sensor is. Raising R "
            "makes the filter trust the sensor less and smooth more; raising Q makes it "
            "react faster to new measurements at the cost of a noisier estimate."
        ),
    },
    {
        "source": "notes/observability",
        "page": 5,
        "text": (
            "A system is observable if its internal state can be reconstructed from its "
            "outputs over a finite time. For a linear system, stack the measurement matrix "
            "and its products with the state transition matrix into the observability "
            "matrix; the system is observable exactly when that matrix has full rank. An "
            "unobservable state cannot be estimated by any filter, no matter how good."
        ),
    },
    {
        "source": "notes/controllability",
        "page": 6,
        "text": (
            "A system is controllable if some input can drive it from any initial state to "
            "any final state in finite time. The controllability matrix stacks the input "
            "matrix and its products with powers of the state matrix; full rank means "
            "every state direction can be reached. Controllability is the dual of "
            "observability."
        ),
    },
    {
        "source": "notes/lqr",
        "page": 7,
        "text": (
            "The linear quadratic regulator finds a feedback gain that minimizes a cost "
            "balancing state deviation against control effort. The weighting matrices Q "
            "and R set that trade-off. The optimal gain comes from solving the algebraic "
            "Riccati equation, and the resulting state-feedback law u = -Kx is optimal for "
            "linear dynamics with quadratic cost. LQR and the Kalman filter are duals: "
            "together they form the LQG controller."
        ),
    },
    {
        "source": "notes/pid",
        "page": 8,
        "text": (
            "A PID controller computes its output from the error between a setpoint and "
            "the measured value. The proportional term reacts to the present error, the "
            "integral term removes steady-state error by accumulating past error, and the "
            "derivative term damps the response by reacting to the rate of change. Too "
            "much integral action causes overshoot and windup; too much derivative action "
            "amplifies sensor noise."
        ),
    },
    {
        "source": "notes/ekf",
        "page": 9,
        "text": (
            "The extended Kalman filter handles nonlinear dynamics or nonlinear "
            "measurements by linearizing them around the current estimate. It replaces "
            "the fixed matrices with Jacobians evaluated at each step, then runs the usual "
            "predict and update equations. Strong nonlinearity or a poor initial estimate "
            "can make the linearization inaccurate and the filter diverge; the unscented "
            "Kalman filter avoids Jacobians by propagating sigma points instead."
        ),
    },
    {
        "source": "notes/stability",
        "page": 10,
        "text": (
            "A continuous-time linear system is asymptotically stable when every "
            "eigenvalue of its state matrix has a negative real part; a discrete-time "
            "system is stable when every eigenvalue lies strictly inside the unit circle. "
            "Lyapunov's method proves stability without solving the dynamics by finding an "
            "energy-like function that always decreases along trajectories."
        ),
    },
]
