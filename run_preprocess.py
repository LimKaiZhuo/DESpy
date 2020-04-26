from own_packages.basic_runs import save_consumer_class_to_file

def selector(**kwargs):
    '''
    This is to read the consumer appliance data in excel into Python, preprocessing the data into a concumer_class Class
    object, and storing them inside a pickle .dat file which can be read later on.

    RUNNING outer GA optimization:
    This must be run through the run_GA.py script since it uses multi-processing to speed the computation. The GA opt
    will save the results in an new results folder in ./results/XXXXX where there will be a GA opt excel file containing
    the optimal allocation.

    name: Name of consumer.dat file that is saved in the ./save/ folder
    excel_name: Path for the excel file containing the consumer consumption data
    sigma: Fraction of population that had signed up for the DSM programme
    solar_path: Path for the solar data excel file.
    '''

    name = kwargs['name']
    excel_name = kwargs['excel_name']
    sigma = kwargs['sigma']
    solar_path = kwargs['solar_path']
    save_consumer_class_to_file(excel_path=excel_name,
                                solar_path=solar_path,
                                name=name,
                                sigma=sigma,
                                k=1,
                                save_path='./save/consumer_class/{}.dat'.format(name))


selector(excel_name='./excel/consumer_individual_data_60c.xlsx',
         name='60c_s100',
         sigma=1, k=1,
         solar_path='./excel/solar_texas.xlsx')
