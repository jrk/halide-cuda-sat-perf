all:
	make -C hl_summed_table
	make -C cuda_summed_table

clean:
	make -C hl_summed_table clean
	make -C cuda_summed_table clean
