% Logistic Regression validation and test code 
% Author: Xiujiao Gao
% Parameters
W = load('w_lr.txt');

% Validation
% X, feature matrix 
X = load(strcat('train0.txt'));
[rows,columns] = size(X);
% get train data , the left will be used as validation data
X(1:ceil(rows*0.95),:)=[];
[rows,columns] = size(X);
xones = ones(rows,1);
X = [xones X];
% T, lable for each x feature,is 1 of 10 vector
T = zeros(rows,10);
T(:,1) = 1;
for i=1:9    
    x = load(strcat('train',num2str(i),'.txt'));
    [rows,columns] = size(x);
    % get training data 
    x(1:ceil(rows*0.95),:)=[];
    [rows,columns] = size(x);
    xones = ones(rows,1);
    X = [X; xones x];
    t = zeros(rows,10);
    t(:,i+1) = 1;
    T = [T;t];
end
A = exp(X*W);
[rows,columns] = size(A);
% Get first Y
rowsum = sum(A,2);
temp = zeros(rows,columns);
for i = 1:columns
    temp(:,i) = rowsum;
end
Y = A./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Lable = zeros(rows,columns);
for i= 1:rows
    Y_Lable(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Lable,T);
validation_err = (sum(sum(E))/2)/rows

% Test 
% X, feature matrix 
X = load(strcat('test0.txt'));
[rows,columns] = size(X);
% get test data 
xones = ones(rows,1);
X = [xones X];
% T, lable for each x feature,is 1 of 10 vector
T = zeros(rows,10);
T(:,1) = 1;
for i=1:9    
    x = load(strcat('test',num2str(i),'.txt'));
    [rows,columns] = size(x);
    xones = ones(rows,1);
    X = [X; xones x];
    t = zeros(rows,10);
    t(:,i+1) = 1;
    T = [T;t];
end        
A = exp(X*W);
[rows,columns] = size(A);
% Get first Y
rowsum = sum(A,2);
temp = zeros(rows,columns);
for i = 1:columns
    temp(:,i) = rowsum;
end
Y = A./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Lable = zeros(rows,columns);
for i= 1:rows
    Y_Lable(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Lable,T);
test_err = (sum(sum(E))/2)/rows

% fid=fopen('classes_lr.txt','w');
% for i=1:rows
%   for j =1:columns
%   fprintf(fid,'%d \t',Y_Lable(i,j));
%   end
%   fprintf(fid,'\n');
% end






