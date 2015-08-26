***Homework Week 2***

$cd Desktop
$cd Git_Repos
$cd Dat8
cd data

#1: What do you think each column means? What do you think each row means?
The order_id - Just a sequential number for each order.
quantity - Not entirely sure.  Perhaps the number of chargeable units per order.
item_name - The name of the items as listed on the Chipotle menu.
choice_description- A computer generated description of the order.
item_price - Total price of the final order.

-------------------------------------------------------------------------------------

#2 - How many orders do there appear to be?

$ tail small chipotle.tsv
1831    1       Carnitas Bowl   [Fresh Tomato Salsa, [Fajita Vegetables, Rice, B
lack Beans, Cheese, Sour Cream, Lettuce]]       $9.25
1831    1       Chips   NULL    $2.15
1831    1       Bottled Water   NULL    $1.50
1832    1       Chicken Soft Tacos      [Fresh Tomato Salsa, [Rice, Cheese, Sour
 Cream]]        $8.75
1832    1       Chips and Guacamole     NULL    $4.45
1833    1       Steak Burrito   [Fresh Tomato Salsa, [Rice, Black Beans, Sour Cr
eam, Cheese, Lettuce, Guacamole]]       $11.75
1833    1       Steak Burrito   [Fresh Tomato Salsa, [Rice, Sour Cream, Cheese,
Lettuce, Guacamole]]    $11.75
1834    1       Chicken Salad Bowl      [Fresh Tomato Salsa, [Fajita Vegetables,
 Pinto Beans, Guacamole, Lettuce]]      $11.25
1834    1       Chicken Salad Bowl      [Fresh Tomato Salsa, [Fajita Vegetables,
 Lettuce]]      $8.75
1834    1       Chicken Salad Bowl      [Fresh Tomato Salsa, [Fajita Vegetables,
 Pinto Beans, Lettuce]] $8.75

***Orders appear to be the first column of numbers. There are 1834 orders in the file.***

-------------------------------------------------------------------------------------

#3: How many lines are in the file?
$wc chipotle tsv
	4623 chipotle.tsv
***There are 4,623 line in the file***

-------------------------------------------------------------------------------------

#4: Which burrito is more popular, steak or chicken?

$ grep -i "steak" chipotle.tsv | wc -l
    706

$ grep -i "chicken" chipotle.tsv | wc -l
   1565
***There are 1565 lines where chicken appears vs 706 for steak. Chicken appears more popular.***

-------------------------------------------------------------------------------------

#5: Do chicken burritos more often have black beans or pinto beans?

$ grep -i "chicken" chipotle.tsv | grep -i "black" | wc -l
    759
$ grep -i "chicken" chipotle.tsv | grep -i "pinto" | wc -l
    265

***chicken burritos are more often ordered with black beans***

-------------------------------------------------------------------------------------

#6: Make a list of all of the CSV or TSV files in the DAT8 repo (using a single command). Think about how wildcard characters can help you with this task

$cd .. (into Dat8)
$ find -name *.csv
./data/airlines.csv
./data/drinks.csv
./data/ufo.csv

$ find -name *.tsv
./data/chipotle.tsv
./data/sms.tsv

-------------------------------------------------------------------------------------

#7: Count the approximate number of occurrences of the word "dictionary" (regardless of case) across all files in the DAT8 repo.

$ grep -ri "dictionary" . * | tr ' ' '\n' |grep -i "dictionary" | wc -l
     34

***34 instances of the word "dictionary"***




