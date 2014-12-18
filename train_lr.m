% Logistic Regression training code 
% Author: Xiujiao Gao
% Parameters
t1 = cputime;
learnrate = 0.0005;
% Error Rate Limit, if big than this, continue
err_limit = 0.001;
% Thresh Limit, if err change bigger than it , continues
thresh_limit = 0.00001;
% iter num
N = 0;
% iter limit,since I use "or" opertion for while, when iter numer is more
% than N_limit, but the error change a lot, it will keep iterating, this
% N_limit will work when the error is already acceptable
N_limit = 700;
% W,first initialize it as ones, then update it during train
%  W = ones(513,10);
% W, initialize it use random numbers between 0 and 1
W = rand(513,10);
fid=fopen('trainout_lr.txt','w');
fid1=fopen('Nerrtrainout_lr.txt','w');
fprintf(fid,'initialize W = \r\n');
for i=1:513
  for j =1:10
  fprintf(fid,'%6.6f \t',W(i,j));
  end
  fprintf(fid,'\n');
end
        
% X, feature matrix 
X = load(strcat('train0.txt'));
[rows,columns] = size(X);
% get train data , the left will be used as validation data
X(ceil(rows*0.95)+1:rows,:)=[];
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
    x(ceil(rows*0.95)+1:rows,:)=[];
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
err = (sum(sum(E))/2)/rows;

% update W
thresh = 1;
while(thresh > thresh_limit | err > err_limit | N < N_limit) % threshhold of updating
    gradientW = zeros(513,10);
    Y_T = Y-T;
   gradientW = X'*Y_T;
% get new W
W = W - learnrate*gradientW;

% compute new Y
A = exp(X*W);
% Get new Y
rowsum = sum(A,2);
temp = zeros(rows,columns);
for i = 1:columns
    temp(:,i) = rowsum;
end
Y = A./temp;
% get max value from Y for each row and the corresponding column number
[y,n] = max(Y');
Y_Label = zeros(rows,columns);
for i= 1:rows
    Y_Label(i,n(i)) = 1;
end
% get error rate
E = xor(Y_Label,T);
err_old = err;
err = (sum(sum(E))/2)/rows
thresh = err_old - err;
if (thresh < 0)
fprintf('Error rate increase from %6.6f to %6.6f, the learning rate maybe too large',err_old,err);
end
fprintf(fid1,' %d  % 6.6f \r\n',N,err);
N = N+1

end     
t2 = cputime;

fprintf(fid,'time = % 6.6f \r\n',t2-t1);
fprintf(fid,'err_limit = % 6.6f \r\n',err_limit);
fprintf(fid,'err_old = % 6.6f \r\n',err_old);
fprintf(fid,'err = % 6.6f \r\n',err);
fprintf(fid,'thresh_limit = % 6.6f \r\n',thresh_limit);
fprintf(fid,'thresh = % 6.6f \r\n',thresh);
fprintf(fid,'learnrate = % 6.6f \r\n',learnrate);
fprintf(fid,'iter times = % d \r\n',N);
fidw = fopen('w_lr.txt');
for i=1:513
  for j =1:10
  fprintf(fid,'%6.6f \t',W(i,j));
  fprintf(fidw,'%6.6f \t',W(i,j));
  end
  fprintf(fid,'\n');
end

for i=1:rows
  for j =1:columns
  fprintf(fid,'%d \t',Y_Label(i,j));
  end
  fprintf(fid,'\n');
end
        
fclose(fid);






