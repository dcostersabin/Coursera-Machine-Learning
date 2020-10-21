function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%



prediction = X * theta;
err = prediction - y;
sq_err = sum(err .^ 2);

cost = (1 / (2 * m)) * sq_err;


% theta_zero(1) = 0;
theta_wbias = theta(2:end);
theta_square = sum(theta_wbias .^ 2);
reg = (lambda / (2 * m)) * theta_square;


J = cost + reg;

grad = (1/m) * (X' * err);

grad(2:end) = grad(2:end) + (lambda / m) * theta_wbias;




% =========================================================================

grad = grad(:);

end
