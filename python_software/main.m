function main(max_k,D1,d1,d2,L1,L2,L3,EIp,Es,F,q)%总函数%
  ls=10;%管道总长度%
  solinit=bvpinit(linspace(0,ls,500),[1 0]);%定义自变量并赋初值，此函数用法见help%
  sol=bvp4c(@fangcheng,@bianjie,solinit);%解微分方程组，第一项为方程组内容，第二项为边界条件，第三项为自变量即迭代的步，此函数用法见help%
  plot(sol.x,sol.y(2,:));%画出结果图%
  hold on;
  x=linspace(0,ls,500);
  max_k=0*x+max_k  %matlab2018b版本原因 新加程序行
  plot(x,max_k,'--r');
  hold on;
  [maxval pos]=max(sol.y(2,:));
  plot(sol.x(pos),maxval,'r*');
  hold off;

   function dyds=fangcheng(x,y)%定义微分方程组%
        EI=gangdu(x);
        dEI=dgangdu(x);
        dyds=[y(2)
              -(dEI*y(2)+F*sin(q-y(1)))/EI];
   end
   function res=bianjie(ya,yb)%定义边界条件%
        res=[ya(1)
             yb(1)-q];
   end
   function EI=gangdu(x)%子函数，用于微分方程组的定义%
        if x<L1
            Ds=D1;
            Is=pi/64*(Ds^4-d1^4);
            EI=Es*Is+EIp;
        elseif x<L1+L2
            Ds=D1-((D1-d2)/L2)*(x-L1);
            Is=pi/64*(Ds^4-d1^4);
            EI=Es*Is+EIp;
        elseif x<L1+L2+L3 
            Ds=d2;
            Is=pi/64*(Ds^4-d1^4);
            EI=Es*Is+EIp;
        else 
            Ds=d1;
            Is=pi/64*(Ds^4-d1^4);
            EI=Es*Is+EIp;
        end
   end
   function dEI=dgangdu(x)%子函数，用于微分方程组的定义%
        if x<L1
            dEI=0;
        elseif x<L1+L2
            Ds=D1-((D1-d2)/L2)*(x-L1);
            dEI=-Es*pi/64*Ds^3*4*(D1-d2)/L2;
        elseif x<L1+L2+L3
            dEI=0;
        else 
            dEI=0;
        end
   end
end