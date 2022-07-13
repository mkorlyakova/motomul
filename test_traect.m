%Обработка данных с датчиков гироскопа и акселерометра. Построение
%траектории. Тестирование алгоритмов обработки данных.

%Эксперименты с гита, 13 штук
Name=[ "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_complex1.csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_complex1(2).csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_000.csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_000_hend.csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_0_2.4_0_0_-2.4(2).csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_0_2.4_0_0_-2.4.csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_2.4_0_-2.4_0.csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_2.4_0.csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_5_0_-5_0.csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_5_0_-5_0(2).csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_2.4_0_0_-2.4_0_0(2).csv"; "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_0_6_0_-6_0.csv";
      "C:\Users\katel\Downloads\motomul-main\motomul-main\treck\new-ag_long_2.4_0_0_-2.4_0_0.csv"
 ];

% Обработка эксперимента
for i=1:13
name=Name(i);
N1 = readmatrix(name);
name=char(name);
j=strfind(name,'\');
name=name(j(length(j))+1:length(name)-4);
name=[name '.png'];

% Удаление строки заголовков (т.к. работа с матрицей)
if isnan(N1(1,1))
    N1(1,:)=[];
end

% Разбор матрицы на данные:
dt=N1(:,2); % время между снятиями показаний
dgx=N1(:,3); % значение отклонения гироскопа по X за время dt
dgy=N1(:,4); % значение отклонения гироскопа по Y за время dt
dgz=N1(:,5); % значение отклонения гироскопа по Z за время dt
dax=N1(:,6); % значение изменения акселерометра по X за время dt
day=N1(:,7); % значение изменения акселерометра по Y за время dt
daz=N1(:,8)-9.87; % значение изменения акселерометра по Z за время dt
% Данные первичной обработки (некорректные?)
vx=N1(:,9);
vy=N1(:,10);
vz=N1(:,11);
x=N1(:,12);
y=N1(:,13);
z=N1(:,14);


% Определение интегральных значений показаний гироскопа
gx=cumsum(dt.*dgx);
gy=cumsum(dt.*dgy);
gz=cumsum(dt.*dgz);

% Определение интегральных значений показаний акселерометра
ax=cumsum(dt.*dax);
ay=cumsum(dt.*day);
az=cumsum(dt.*daz);


% Построение траектории по гироскопу и акселерометру:
R=sqrt(dax.^2+day.^2);
X=cumsum(dt.*R.*cos(gz));
Y=cumsum(dt.*R.*sin(gz));


%Отрисовка
subplot(2,2,1)
hold off;
plot(ax)
hold on; plot(ay); plot(az)
legend('ax','ay','az')
title([name ' a'])
subplot(2,2,2)
hold off;
plot(gx)
hold on; plot(gy); plot(gz)
legend('gx','gy','gz')
title('g')
subplot(2,2,[3 4]); 
plot(X,Y)
title('Путь')
xlabel('x')
ylabel('y')

%Сохранение
print('-f1',name,'-dpng')
end