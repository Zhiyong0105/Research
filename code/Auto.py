import io
import sys

def inti(mx,my):
    print("#include <stdio.h>")
    print("#include <stdlib.h>")
    print("#include <string.h>")
    print("#include <math.h>")
    print("#include <immintrin.h>")
    print(" #include \"function.h\" ")
    
    print('\n')
    # print("void applywavemx2_avx_{}x{}(int m, double *V, double *G, int ldv, int ldg, int g, int i)".format(mx,my))
    print("void applywavemx2_avx_auto(int m, double *V, double *G, int ldv, int ldg, int g, int i)")
    print("{")
    ct = 0
    for i in range(mx+1):
        print("  double *restrict v{0} = &V[(g+{0}) * ldv];".format(i))
        ct += 1
    for j in range(1,my):
        print("  double *restrict v{0} = &V[(g-{1}) * ldv];".format(ct,j))
        ct += 1
    
        
    str_startg = "__m512d  g1 "
    str_starts = "__m512d  s1 "
    str_startv =  "__m512d  v0_vec "
    for i in range(2,mx*my+1):
        str_startg += ",g{} ".format(i)
        str_starts += ",s{} ".format(i)
        
    str_startg += "; "
    str_starts += "; "
    print(str_startg)
    print(str_starts)
    for i in range(1,mx+1):
        str_startv += ",v{}_vec".format(i)
    str_startv += "; "
    print(str_startv)
    print('\n')
    print("int m_iter = m / 8;")
    print("int m_left = m % 8;")
    print('\n')
    print("__m512d  tmp_vec;")
    for j in range(my):
        for i in range(mx):
            gname = "g{}".format(j*mx+i+1)
            # Gaddr = "G[2 * (i + {}) + (g + {}) * ldg ]".format(j,i-j)
            # print("  G[2 * (i + {}) + (g + {}) * ldg ]".format(j,i-j))
            # Saddr = "G[2 * (i + {}) + (g + {}) * ldg +1]".format(j,i-j)
            # print("  G[2 * (i + {}) + (g + {}) * ldg +1]".format(j,i-j))
            # gSet = "g{}=_mm512_set1_pd(G[2 * (i+{}) + (g+{}) * ldg]); ".format(j*mx+i+1,j,i-j)
            print("  g{}=_mm512_set1_pd(G[2 * (i+{}) + (g+{}) * ldg]); ".format(j*mx+i+1,j,i-j))
            # sSet = "s{}=_mm512_set1_pd(G[2 * (i+{}) + (g+{}) * ldg+1]); ".format(j*mx+i+1,j,i-j) 
            print("  s{}=_mm512_set1_pd(G[2 * (i+{}) + (g+{}) * ldg+1]); ".format(j*mx+i+1,j,i-j) )
            # print(gSet)
            # print(sSet)
            print('\n')
    str_for = " for (int j = 0; j < m_iter; j++"
    for j in range (mx+my):
        str_for += ",v{} += 8".format(j)
       
    str_for += ")"
    print(str_for) 
    print("{")
    for j in range (mx+1):
         print("  v{}_vec = _mm512_loadu_pd(v{});".format(j,j))
    varlist =[]
    collist =[]
    for i in range(mx+1):
        collist.append("v{}".format(i))  
    # print(collist)    
    for i in range(mx+1):
        varlist.append("v{}".format(i))
    # print(varlist)
    count = 1
    for j in range(my):
        if j>0 :
            print("  _mm512_storeu_pd({0}, {0}_vec);".format(collist[-1]))
            collist.pop()
            collist = ["v{}".format(mx+j)]+ collist[0:mx]
            print("  {}_vec = _mm512_loadu_pd(v{});".format(varlist[0],mx+j))
        for i in  range(mx): 
            
            a = varlist[i]
            b = varlist[i+1]
            str_app="  tmp_vec = {0}_vec;\n  {0}_vec= _mm512_add_pd(_mm512_mul_pd(g{2}, tmp_vec), _mm512_mul_pd(s{2}, {1}_vec));\n  {1}_vec = _mm512_sub_pd(_mm512_mul_pd(g{2}, {1}_vec), _mm512_mul_pd(s{2}, tmp_vec));".format(a,b,count)
            count +=1
            print(str_app)
            print('\n')
        if j< my-1:
            varlist = [varlist[-1]]   +varlist[0:mx] 
    # print(collist)
    for i in range( len(collist)):
        print("  _mm512_storeu_pd({}, {}_vec);".format(collist[i],varlist[i]))
       # print(varlist)
    print("}")
    
    print("  if (m_left > 0)")
    print("{")
    print(" __mmask8 mask = (__mmask8)(255 >> (8 - m_left));")
    print('\n')
    
    vec_mask_list =[]
    vec_mask_list1 = []
    for i in range(mx+1):
        vec_mask_list.append("v{}".format(i))
        vec_mask_list1.append("v{}".format(i))
    # print(vec_mask_list)
    count1 =1
    for i in range(mx+1):
        print("  {0}_vec = _mm512_maskz_loadu_pd(mask, {0});".format(vec_mask_list[i]))
    print('\n')
    for j in range(my):
        if j>0 :
            print("  _mm512_mask_storeu_pd({0}, mask, {0}_vec);".format(vec_mask_list1[-1]))
            vec_mask_list1.pop()
            vec_mask_list1 = ["v{}".format(mx+j)]+ vec_mask_list1[0:mx]
            print("  {}_vec = _mm512_maskz_loadu_pd(mask, v{});".format(varlist[0],mx+j))
        for i in range(mx):
            a = vec_mask_list[i]
            b = vec_mask_list[i+1]
            print("  tmp_vec = {}_vec;".format(a))
            print("  {0}_vec = _mm512_add_pd(_mm512_mul_pd(g{2}, tmp_vec), _mm512_mul_pd({1}_vec, {0}_vec));".format(a,b,count1))
            print("  {0}_vec = _mm512_sub_pd(_mm512_mul_pd(g{1}, {0}_vec), _mm512_mul_pd(s{1}, tmp_vec));".format(b,count1))
            count1 += 1
            print('\n')
        if j<my-1:
            vec_mask_list = [vec_mask_list[-1]] + vec_mask_list[0:mx]
        
    for i in range(len(vec_mask_list)):
        print("  _mm512_mask_storeu_pd({}, mask, {}_vec);".format(vec_mask_list1[i],vec_mask_list[i]))
            
    print("}")
    print("}")
    
                
    
original_stdout = sys.stdout
fake_stdout = io.StringIO()
sys.stdout = fake_stdout
args = sys.argv


inti(args[1], args[2])


with open("function.c", "w") as f:
    f.write(fake_stdout.getvalue())


sys.stdout = original_stdout
