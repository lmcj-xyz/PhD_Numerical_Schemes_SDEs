for i in range(10):
    exec(open('error.py').read())
    # If you are using this script, go and comment the last line of
    # error.py where plt.plot() is invoked
    plt.savefig('beta04375-'+str(i)+'.png')
    plt.clf()
    print('rate = ', rate_strong)
    file = open("rates.txt", "a")
    file.write(str(rate_strong))
    file.write("\n")
    file.close()
