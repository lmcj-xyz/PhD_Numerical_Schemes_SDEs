for i in range(40):
    exec(open('error_mv.py').read())
    # If you are using this script, go and comment the last line of
    # error.py where plt.plot() is invoked
    #plt.savefig('beta04375-'+str(i)+'.png')
    #plt.clf()
    print('rate = ', rate_strong)
    file = open("rates_mv.txt", "a")
    file.write(str(rate_strong))
    file.write("\n")
    file.close()
