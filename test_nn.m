%  Neural network training code 
% Author: Xiujiao Gao
% Parameters

t1 = cputime;
%  get weight matrix
W1 = load('W1.txt');   
W2 = load('W2.txt');  

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
% Get Y
A = X*W1;
[Arows,Acolumns] = size(A);
Z = zeros(Arows,Acolumns);
for i = 1:Arows
    for j = 1:Acolumns
        Z(i,j) = tanh(A(i,j));
    end
end
z = ones(Arows,1)*1.0;
Z = [z Z];
R = Z*W2;
% compute  Y
Sig = exp(R);
[Sigrows,Sigcolumns] = size(Sig);
rowsum = sum(Sig,2);
temp = zeros(Sigrows,Sigcolumns);
for i = 1:Sigcolumns
    temp(:,i) = rowsum;
end
Y = Sig./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Lable = zeros(Sigrows,Sigcolumns);
for i= 1:Sigrows
    Y_Lable(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Lable,T);
validerr = (sum(sum(E))/2)/Sigrows

% Test part
% X, feature matrix 
X = load(strcat('test0.txt'));
[Xrows,Xcolumns] = size(X);
xones = ones(Xrows,1);
X = [xones X];
% T, lable for each x feature,is 1 of 10 vector
T = zeros(Xrows,10);
T(:,1) = 1;
for i=1:9    
    x = load(strcat('test',num2str(i),'.txt'));
    [xrows,xcolumns] = size(x);
    xones = ones(xrows,1);
    X = [X; xones x];
    t = zeros(xrows,10);
    t(:,i+1) = 1;
    T = [T;t];
end

% Get Y
A = X*W1;
[Arows,Acolumns] = size(A);
Z = zeros(Arows,Acolumns);
for i = 1:Arows
    for j = 1:Acolumns
        Z(i,j) = tanh(A(i,j));
    end
end
z = ones(Arows,1)*1.0;
Z = [z Z];
R = Z*W2;
% compute  Y
Sig = exp(R);
[Sigrows,Sigcolumns] = size(Sig);
rowsum = sum(Sig,2);
temp = zeros(Sigrows,Sigcolumns);
for i = 1:Sigcolumns
    temp(:,i) = rowsum;
end
Y = Sig./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Lable = zeros(Sigrows,Sigcolumns);
for i= 1:Sigrows
    Y_Lable(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Lable,T);
testerr = (sum(sum(E))/2)/Sigrows

fid = fopen('class_nn.txt','W');
for i=1:Sigrows
  for j =1:Sigcolumns
  fprintf(fid,'%d \t',Y_Lable(i,j));
  end
  fprintf(fid,'\n');
end
        
fclose(fid);






