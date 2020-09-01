function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%        
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
#theta1 = 25 * 401
#theta2 = 10 * 26

#computing cost

X_bias = [ones(m, 1) X];
z2 = X_bias * Theta1';
a2 = sigmoid(z2);

mm = size(a2, 1);
a2_bias = [ones(mm, 1) a2];
z3 = a2_bias * Theta2';
hypoThesis = sigmoid(z3);  # a3 = hypoThesis

for i=1:m
  hTheta = hypoThesis(i, :);  
  yy = zeros(1, num_labels);
  yy(y(i)) = 1;
  J = J + ( -yy * log(hTheta') - (1-yy) * log(1-hTheta') );
endfor

J = J/m;

#regularization

theta1WoBias = Theta1(:, 2:end);
theta2WoBias = Theta2(:, 2:end);


theta1Sum = sum(sum(theta1WoBias .^2));
theta2Sum = sum(sum(theta2WoBias .^2));
J = J + (lambda * (theta1Sum + theta2Sum) )/ (2*m);

#regularization ends

#BackPropagation

for i = 1:m
  hTheta = hypoThesis(i, :);  
  yy = zeros(1, num_labels);
  yy(y(i)) = 1;
  delta3 = hTheta - yy;
  delta2 = (delta3 * Theta2)  .* sigmoidGradient([1 z2(i,:)]);
  
  delta2 = delta2(2:end);
  
  Theta1_grad = Theta1_grad + (delta2' * X_bias(i,:));
  Theta2_grad = Theta2_grad + (delta3' * a2_bias(i,:));
  
endfor

Theta1_grad = Theta1_grad / m;
Theta2_grad = Theta2_grad /  m;

#regularization

theta1_first_col = Theta1_grad(:,1);
theta2_first_col = Theta2_grad(:,1);

tempTheta1 = Theta1 * lambda / m;
tempTheta2 = Theta2 * lambda / m;

Theta1_grad = Theta1_grad + tempTheta1;
Theta2_grad = Theta2_grad + tempTheta2;

Theta1_grad(:,1) = theta1_first_col;
Theta2_grad(:,1) = theta2_first_col;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
