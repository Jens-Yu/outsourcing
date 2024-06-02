function main(max_k,D1,d1,d2,L1,L2,L3,EIp,Es,F,q)%�ܺ���%
  ls=10;%�ܵ��ܳ���%
  solinit=bvpinit(linspace(0,ls,500),[1 0]);%�����Ա���������ֵ���˺����÷���help%
  sol=bvp4c(@fangcheng,@bianjie,solinit);%��΢�ַ����飬��һ��Ϊ���������ݣ��ڶ���Ϊ�߽�������������Ϊ�Ա����������Ĳ����˺����÷���help%
  plot(sol.x,sol.y(2,:));%�������ͼ%
  hold on;
  x=linspace(0,ls,500);
  max_k=0*x+max_k  %matlab2018b�汾ԭ�� �¼ӳ�����
  plot(x,max_k,'--r');
  hold on;
  [maxval pos]=max(sol.y(2,:));
  plot(sol.x(pos),maxval,'r*');
  hold off;

   function dyds=fangcheng(x,y)%����΢�ַ�����%
        EI=gangdu(x);
        dEI=dgangdu(x);
        dyds=[y(2)
              -(dEI*y(2)+F*sin(q-y(1)))/EI];
   end
   function res=bianjie(ya,yb)%����߽�����%
        res=[ya(1)
             yb(1)-q];
   end
   function EI=gangdu(x)%�Ӻ���������΢�ַ�����Ķ���%
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
   function dEI=dgangdu(x)%�Ӻ���������΢�ַ�����Ķ���%
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