# trajectory*bitransformer*

This is the code of our approach for trajectory prediction using transformer. Our work is described here **link to paper**

## Requirements

- Pytorch
- Pandas
- Numpy
  > Our docker image can be find here

## Data setup

The dataset should be in the /data folder and have the following structure

    - data
      - data_trajpred
        - datasetname
            - visual_data
            - pos_data_train.db
            - pos_data_val.db
            - pos_data.db

## Usage

### Individual seq2seq bimodal transformer

<figure>
<center><img src="https://raw.githubusercontent.com/schockschock/trajectory_bitransformer_/main/img/trajectory_bimodal_transformer.drawio%20(2).png" title="Alternative text" width="1300"/>
<figcaption align = "center"> Figure 1 : Diagram of our model</figcaption>
</center>
</figure>

To train the bimodal transformer just use the notebook _train_bimodal_transformer.ipynb_

### Individual seq2seq cross attention transformer

<figure>
<center><img src="https://raw.githubusercontent.com/schockschock/trajectory_bitransformer_/main/img/Trajectory_cross_attention_transformer.drawio.png" title="Alternative text" width="1300"/>
<figcaption align = "center"> Figure 1 : Diagram of our model</figcaption>
</center>
</figure>

To train the bimodal transformer just use the notebook _train_cross_attention_transformer.ipynb_

## TODO

- [x] Try bimodal
- [x] Try cross attention
- [ ] Try with sublayer connection (residual connection)

# Thanks

Thanks to our teacher Mr Hazem Wannouss and to Mayssa ZAIER for giving us the opportunity to tempt our own approach on this interesting usecase.
None of this would have been possible without you two.
