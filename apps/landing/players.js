const transition = document.querySelector(".page-transition");

document.querySelectorAll("a").forEach(link => {

    link.addEventListener("click", e => {

        const href = link.getAttribute("href");

        if(href && !href.startsWith("#")) {

            e.preventDefault();

            transition.classList.add("active");

            setTimeout(() => {

                window.location.href = href;

            }, 500);
        }

    });

});

const players = [
  { name:"MS DHONI", country:"India", role:"Wicket Keeper", basePrice:25000000 },
  { name:"ROHIT SHARMA", country:"India", role:"Batter", basePrice:25000000 },
  { name:"VIRAT KOHLI", country:"India", role:"Batter", basePrice:25000000 },
  { name:"R ASHWIN", country:"India", role:"All-Rounder", basePrice:25000000 },
  { name:"TRENT BOULT", country:"New Zealand", role:"Bowler", basePrice:25000000 },
  { name:"PAT CUMMINS", country:"Australia", role:"Bowler", basePrice:25000000 },
  { name:"KAGISO RABADA", country:"South Africa", role:"Bowler", basePrice:25000000 },
  { name:"DAVID WARNER", country:"Australia", role:"Batter", basePrice:25000000 },
  { name:"JASPRIT BUMRAH", country:"India", role:"Bowler", basePrice:25000000 },
  { name:"QUINTON DE KOCK", country:"South Africa", role:"Wicket Keeper", basePrice:25000000 },
  { name:"RAVINDRA JADEJA", country:"India", role:"All-Rounder", basePrice:25000000 },
  { name:"MITCHELL STARC", country:"Australia", role:"Bowler", basePrice:25000000 },
  { name:"SHREYAS IYER", country:"India", role:"Batter", basePrice:25000000 },
  { name:"KANE WILLIAMSON", country:"New Zealand", role:"Batter", basePrice:25000000 },
  { name:"MOHAMMAD SHAMI", country:"India", role:"Bowler", basePrice:25000000 },
  { name:"AJINKYA RAHANE", country:"India", role:"Batter", basePrice:20000000 },
  { name:"PRITHVI SHAW", country:"India", role:"Batter", basePrice:15000000 },
  { name:"NITISH RANA", country:"India", role:"Batter", basePrice:10000000 },
  { name:"DEVDUTT PADIKKAL", country:"India", role:"Batter", basePrice:10000000 },
  { name:"DEWALD BREVIS", country:"South Africa", role:"Batter", basePrice:15000000 },
  { name:"JITESH SHARMA", country:"India", role:"Wicket Keeper", basePrice:15000000 },
  { name:"RIYAN PARAG", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"SHASHANK SINGH", country:"India", role:"Batter", basePrice:5000000 },
  { name:"SHIMRON HETMYER", country:"West Indies", role:"Batter", basePrice:20000000 },
  { name:"SUYASH PRABHUDESAI", country:"India", role:"Batter", basePrice:5000000 },
  { name:"RAHUL TRIPATHI", country:"India", role:"Batter", basePrice:10000000 },
  { name:"RAJAT PATIDAR", country:"India", role:"Batter", basePrice:20000000 },
  { name:"MAYANK AGGARWAL", country:"India", role:"Batter", basePrice:10000000 },
  { name:"ANMOLPREET SINGH", country:"India", role:"Batter", basePrice:5000000 },
  { name:"YASHASVI JAISWAL", country:"India", role:"Batter", basePrice:20000000 },
  { name:"RILEE ROSSOUW", country:"South Africa", role:"Batter", basePrice:15000000 },
  { name:"SIMARJEET SINGH", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"MUKESH KUMAR", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"JOSHUA LITTLE", country:"Ireland", role:"Bowler", basePrice:10000000 },
  { name:"NOOR AHMED", country:"Afghanistan", role:"Bowler", basePrice:15000000 },
  { name:"SPENCER JHONSON", country:"Australia", role:"Bowler", basePrice:10000000 },
  { name:"MATT HENREY", country:"New Zealand", role:"Bowler", basePrice:10000000 },
  { name:"CHETAN SAKARIA", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"VAIBHAV ARORA", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"NAVEEN UL HAQ", country:"Afghanistan", role:"Bowler", basePrice:10000000 },
  { name:"SHIVAM MAVI", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"NUWAN TUSHARA", country:"Sri Lanka", role:"Bowler", basePrice:10000000 },
  { name:"LUKE WOOD", country:"England", role:"Bowler", basePrice:10000000 },
  { name:"JAYDEV UNADKAT", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"VIJAYAKANTH VIYASKANTH", country:"Sri Lanka", role:"Bowler", basePrice:2000000 },
  { name:"AKASH SINGH", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"TOM CURRAN", country:"England", role:"Bowler", basePrice:10000000 },
  { name:"AKASH DEEP", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"NANDRE BURGER", country:"South Africa", role:"Bowler", basePrice:10000000 },
  { name:"KESHAV MAHARAJ", country:"South Africa", role:"Bowler", basePrice:10000000 },
  { name:"HARSHAL PATEL", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"RACHIN RAVINDRA", country:"New Zealand", role:"All-Rounder", basePrice:15000000 },
  { name:"AXAR PATEL", country:"India", role:"All-Rounder", basePrice:20000000 },
  { name:"MITCHELL MARSH", country:"New Zealand", role:"All-Rounder", basePrice:20000000 },
  { name:"SUNIL NARINE", country:"West Indies", role:"All-Rounder", basePrice:20000000 },
  { name:"MAHIPAL LOMROR", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"SHAHRUKH KHAN", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"VIJAY SHANKAR", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"VENKATESH IYER", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"MARCUS STOINIS", country:"Australia", role:"All-Rounder", basePrice:20000000 },
  { name:"KYLE MAYERS", country:"West Indies", role:"All-Rounder", basePrice:10000000 },
  { name:"DAVID WILLEY", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"HARDIK PANDYA", country:"India", role:"All-Rounder", basePrice:20000000 },
  { name:"SAM CURRAN", country:"England", role:"All-Rounder", basePrice:20000000 },
  { name:"GLENN MAXWELL", country:"Australia", role:"All-Rounder", basePrice:20000000 },
  { name:"ABHISHEK SHARMA", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"ASHUTOSH SHARMA", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"NITISH KUMAR REDDY", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"CAMERON GREEN", country:"Australia", role:"All-Rounder", basePrice:15000000 },
  { name:"WASHINGTON SUNDAR", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"SHARDUL THAKUR", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"LALIT YADAV", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"AZMATULLAH OMARZAI", country:"Afghanistan", role:"All-Rounder", basePrice:10000000 },
  { name:"ANDRE RUSSEL", country:"West Indies", role:"All-Rounder", basePrice:20000000 },
  { name:"RAMANDEEP SINGH", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"KRUNAL PANDYA", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"NEHAL WADHERA", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"GERALD COETZEE", country:"South Africa", role:"All-Rounder", basePrice:10000000 },
  { name:"CHRIS WOAKES", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"SIKANDAR RAZA", country:"Zimbabwe", role:"All-Rounder", basePrice:2000000 },
  { name:"SWAPNIL SINGH", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"WANINDU HASARANGA", country:"Sri Lanka", role:"All-Rounder", basePrice:10000000 },
  { name:"MARCO JANSEN", country:"South Africa", role:"All-Rounder", basePrice:10000000 },
  { name:"MOEEN ALI", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"ANUKUL ROY", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"KRISHNAPPA GOWTHAM", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"DEEPAK HOODA", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"MOHAMMAD NABI", country:"Afghanistan", role:"All-Rounder", basePrice:10000000 },
  { name:"ROMARIO SHEPHERD", country:"West Indies", role:"All-Rounder", basePrice:10000000 },
  { name:"RISHI DHAWAN", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"KARN SHARMA", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"WILL JACKS", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"RAHUL TEWATIA", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"BEN STOKES", country:"England", role:"All-Rounder", basePrice:15000000 },
  { name:"PAWAN NEGI", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"SHIVAM DUBE", country:"India", role:"All-Rounder", basePrice:15000000 },
  { name:"SHAKIB AL HASSAN", country:"Bangladesh", role:"All-Rounder", basePrice:10000000 },
  { name:"ODEAN SMITH", country:"West Indies", role:"All-Rounder", basePrice:5000000 },
  { name:"LIAM LIVINGSTONE", country:"England", role:"All-Rounder", basePrice:15000000 },
  { name:"KEEMO PAUL", country:"West Indies", role:"All-Rounder", basePrice:5000000 },
  { name:"SHERFANE RUTHERFORD", country:"West Indies", role:"All-Rounder", basePrice:5000000 },
  { name:"JAYANT YADAV", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"KEDAR JADHAV", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"COLIN MUNRO", country:"New Zealand", role:"All-Rounder", basePrice:10000000 },
  { name:"DARYL MITCHELL", country:"New Zealand", role:"All-Rounder", basePrice:10000000 },
  { name:"HARPREET BRAR", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"MOISES HENRIQUES", country:"Australia", role:"All-Rounder", basePrice:15000000 },
  { name:"MITCHELL SANTNER", country:"New Zealand", role:"All-Rounder", basePrice:15000000 },
  { name:"JASON HOLDER", country:"West Indies", role:"All-Rounder", basePrice:10000000 },
  { name:"DANIEL SAMS", country:"Australia", role:"All-Rounder", basePrice:5000000 },
  { name:"PRAVIN DUBE", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"CHRIS JORDAN", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"ASHTON TURNER", country:"Australia", role:"All-Rounder", basePrice:5000000 },
  { name:"HARRY BROOK", country:"England", role:"Batter", basePrice:10000000 },
  { name:"MANISH PANDEY", country:"India", role:"Batter", basePrice:10000000 },
  { name:"CHRIS LYNN", country:"Australia", role:"Batter", basePrice:10000000 },
  { name:"MANDEEP SINGH", country:"India", role:"Batter", basePrice:2000000 },
  { name:"HANUMA VIHARI", country:"India", role:"Batter", basePrice:2000000 },
  { name:"EVIN LEWIS", country:"West Indies", role:"Batter", basePrice:5000000 },
  { name:"DAVID MILLER", country:"South Africa", role:"Batter", basePrice:10000000 },
  { name:"SHUBMAN GILL", country:"India", role:"Batter", basePrice:20000000 },
  { name:"GURKEERAT SINGH", country:"India", role:"Batter", basePrice:5000000 },
  { name:"MARNUS LABUSHANE", country:"Australia", role:"Batter", basePrice:5000000 },
  { name:"ALEX HALES", country:"England", role:"Batter", basePrice:10000000 },
  { name:"ROVMAN POWELL", country:"West Indies", role:"Batter", basePrice:10000000 },
  { name:"RINKU SINGH", country:"India", role:"Batter", basePrice:10000000 },
  { name:"MANAN VOHRA", country:"India", role:"Batter", basePrice:5000000 },
  { name:"JASON ROY", country:"England", role:"Batter", basePrice:10000000 },
  { name:"SURYAKUMAR YADAV", country:"India", role:"Batter", basePrice:20000000 },
  { name:"TRAVIS HEAD", country:"Australia", role:"Batter", basePrice:20000000 },
  { name:"ABDUL SAMAD", country:"India", role:"Batter", basePrice:5000000 },
  { name:"DEVON CONWAY", country:"New Zealand", role:"Batter", basePrice:15000000 },
  { name:"NAJIBULLAH ZADRAN", country:"Afghanistan", role:"Batter", basePrice:5000000 },
  { name:"SANDEEP SHARMA", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"ALZARRI JOSEPH", country:"West Indies", role:"Bowler", basePrice:10000000 },
  { name:"BHUVANESHWAR KUMAR", country:"India", role:"Bowler", basePrice:20000000 },
  { name:"T NATARAJAN", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"AKASH MADHWAL", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"SHREYAS GOPAL", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"RAVI BISHNOI", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"HARSHIT RANA", country:"India", role:"Bowler", basePrice:20000000 },
  { name:"DUSHMANTA CHAMEERA", country:"Sri Lanka", role:"Bowler", basePrice:10000000 },
  { name:"KARTIK TYAGI", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"UMESH YADAV", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"RASHID KHAN", country:"Afghanistan", role:"Bowler", basePrice:20000000 },
  { name:"MOHIT SHARMA", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"ANRICH NORTJE", country:"South Africa", role:"Bowler", basePrice:20000000 },
  { name:"KULDEEP YADAV", country:"India", role:"Bowler", basePrice:20000000 },
  { name:"ISHANT SHARMA", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"KHALEEL AHMED", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"MAHEESH THEEKSHANA", country:"Sri Lanka", role:"Bowler", basePrice:15000000 },
  { name:"YUZI CHAHAL", country:"India", role:"Bowler", basePrice:20000000 },
  { name:"TUSHAR DESHPANDE", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"MUSTAFIZUR RAHMAN", country:"Bangladesh", role:"Bowler", basePrice:10000000 },
  { name:"JHYE RICHARDSON", country:"Australia", role:"Bowler", basePrice:10000000 },
  { name:"SAI KISHORE", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"SANDEEP WARRIER", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"SUYASH SHARMA", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"VARUN CHAKRAVARTHY", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"SHAMAR JOSEPH", country:"West Indies", role:"Bowler", basePrice:10000000 },
  { name:"MOHSIN KHAN", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"MAYANK YADAV", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"UMRAN MALIK", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"MAYANK MARKANDE", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"LOCKIE FERGUSON", country:"New Zealand", role:"Bowler", basePrice:15000000 },
  { name:"VYSHAK VIJAY KUMAR", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"MOHAMMAD SIRAJ", country:"India", role:"Bowler", basePrice:20000000 },
  { name:"AVESH KHAN", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"ARSHDEEP SINGH", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"RAHUL CHAHAR", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"DEEPAK CHAHAR", country:"India", role:"Bowler", basePrice:15000000 },
  { name:"NATHAN ELLIS", country:"Australia", role:"Bowler", basePrice:10000000 },
  { name:"NAVDEEP SAINI", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"KULDEEP SEN", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"YASH DAYAL", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"REECE TOPLY", country:"England", role:"Bowler", basePrice:15000000 },
  { name:"HIMANSHU SHARMA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"FAZALHAQ FAROOQI", country:"Afghanistan", role:"Bowler", basePrice:5000000 },
  { name:"M SIDDHARTH", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"YASH THAKUR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"MUKESH CHOUDHARY", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"MATHEESHA PATHIRANA", country:"Sri Lanka", role:"Bowler", basePrice:15000000 },
  { name:"JOFRA ARCHER", country:"England", role:"Bowler", basePrice:15000000 },
  { name:"DAVID MALAN", country:"England", role:"Batter", basePrice:10000000 },
  { name:"RUTURAJ GAIKWAD", country:"India", role:"Batter", basePrice:20000000 },
  { name:"MANOJ TIWARY", country:"India", role:"Batter", basePrice:5000000 },
  { name:"FAF DU PLESISSIS", country:"South Africa", role:"Batter", basePrice:15000000 },
  { name:"KARUN NAIR", country:"India", role:"Batter", basePrice:10000000 },
  { name:"MARTIN GUPTIL", country:"New Zealand", role:"Batter", basePrice:10000000 },
  { name:"ISHANK JAGGI", country:"India", role:"Batter", basePrice:2000000 },
  { name:"ROBIN UTHAPPA", country:"India", role:"Batter", basePrice:10000000 },
  { name:"STEVE SMITH", country:"Australia", role:"Batter", basePrice:10000000 },
  { name:"SHIKHAR DHAWAN", country:"India", role:"Batter", basePrice:15000000 },
  { name:"RASSIE VAN DER DUSSEN", country:"South Africa", role:"Batter", basePrice:10000000 },
  { name:"PRIYAM GARG", country:"India", role:"Batter", basePrice:2000000 },
  { name:"SHAIK RASHEED", country:"India", role:"Batter", basePrice:5000000 },
  { name:"SAMEER RIZVI", country:"India", role:"Batter", basePrice:10000000 },
  { name:"AVINASH RAOARAVELLY", country:"India", role:"Batter", basePrice:2000000 },
  { name:"YASH DHULL", country:"India", role:"Batter", basePrice:5000000 },
  { name:"JAKE FRASER-MCGURK", country:"Australia", role:"Batter", basePrice:10000000 },
  { name:"SHAI HOPE", country:"West Indies", role:"Batter", basePrice:10000000 },
  { name:"ANGKRISH RAGHUVANSHI", country:"India", role:"Batter", basePrice:10000000 },
  { name:"AIDEN MARKRAM", country:"South Africa", role:"Batter", basePrice:15000000 },
  { name:"TIM DAVID", country:"Australia", role:"Batter", basePrice:15000000 },
  { name:"N. TILAK VARMA", country:"India", role:"Batter", basePrice:15000000 },
  { name:"HARPREET BHATIA", country:"India", role:"Batter", basePrice:2000000 },
  { name:"SHUBHAM DUBEY", country:"India", role:"Batter", basePrice:2000000 },
  { name:"SAURAV CHAUHAN", country:"India", role:"Batter", basePrice:2000000 },
  { name:"SWASTIK CHHIKARA", country:"India", role:"Batter", basePrice:5000000 },
  { name:"MARK WOOD", country:"England", role:"Bowler", basePrice:10000000 },
  { name:"KULWANT KHEJROLIA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"MUJEEB UR RAHMAN", country:"Afghanistan", role:"Bowler", basePrice:10000000 },
  { name:"SIDDHARTH KAUL", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"PRASIDH KRISHNA", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"TIM SOUTHEE", country:"New Zealand", role:"Bowler", basePrice:10000000 },
  { name:"LUNGISANI NGIDI", country:"South Africa", role:"Bowler", basePrice:10000000 },
  { name:"ANIKET CHOUDHARY", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"JOSH HAZELWOOD", country:"Australia", role:"Bowler", basePrice:20000000 },
  { name:"ISH SODHI", country:"New Zealand", role:"Bowler", basePrice:10000000 },
  { name:"JAGADEESHA SUCHIT", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"RAHUL SHUKLA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"JASON BEHRENDOFF", country:"Australia", role:"Bowler", basePrice:5000000 },
  { name:"SHAHBAZ NADEEM", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"KAMLESH NAGARKOTI", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"PIYUSH CHAWLA", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"ADIL RASHID", country:"England", role:"Bowler", basePrice:10000000 },
  { name:"ANKIT SINGH RAJPOOT", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"MURUGAN ASHWIN", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"BASIL THAMPI", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"NATHAN COULTER-NILE", country:"Australia", role:"Bowler", basePrice:10000000 },
  { name:"RAHUL SHARMA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"AMIT MISHRA", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"ADAM ZAMPA", country:"Australia", role:"Bowler", basePrice:10000000 },
  { name:"RICHARD GLEESON", country:"England", role:"Bowler", basePrice:10000000 },
  { name:"RAJVARDHAN HANGARGKER", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"RASIKH DAR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"MANAV SUTHAR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"SUSHANT MISHRA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"SAKIB HUSSAIN", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"KUMAR KARTIKEYA SINGH", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"DILSHAN MADHUSHANKA", country:"Sri Lanka", role:"Bowler", basePrice:5000000 },
  { name:"RAJAN KUMAR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"SHAHBAZ AHMED", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"ARAVELLY AVANISH", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"ANUJ RAWAT", country:"India", role:"Wicket Keeper", basePrice:5000000 },
  { name:"ABISHEK POREL", country:"India", role:"Wicket Keeper", basePrice:10000000 },
  { name:"RICKY BHUI", country:"India", role:"Wicket Keeper", basePrice:5000000 },
  { name:"KUMAR KUSHAGRA", country:"Sri Lanka", role:"Wicket Keeper", basePrice:5000000 },
  { name:"RAHMANULLAH GURBAZ", country:"Afghanistan", role:"Wicket Keeper", basePrice:10000000 },
  { name:"SRIKAR BHARAT", country:"India", role:"Wicket Keeper", basePrice:5000000 },
  { name:"ROBIN MINZ", country:"India", role:"Wicket Keeper", basePrice:5000000 },
  { name:"MATTHEW WADE", country:"Australia", role:"Wicket Keeper", basePrice:10000000 },
  { name:"WRIDDHIMAN SAHA", country:"India", role:"Wicket Keeper", basePrice:10000000 },
  { name:"UPENDRA YADAV", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"HEINRICH KLAASEN", country:"South Africa", role:"Wicket Keeper", basePrice:20000000 },
  { name:"NICHOLAS POORAN", country:"West Indies", role:"Wicket Keeper", basePrice:20000000 },
  { name:"KL RAHUL", country:"India", role:"Wicket Keeper", basePrice:20000000 },
  { name:"JOS BUTTLER", country:"England", role:"Wicket Keeper", basePrice:20000000 },
  { name:"SANJU SAMSON", country:"India", role:"Wicket Keeper", basePrice:20000000 },
  { name:"KUNAL SINGH RATHORE", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"TOM KOHLER-CADMORE", country:"England", role:"Wicket Keeper", basePrice:5000000 },
  { name:"JONNY BAIRSTOW", country:"England", role:"Wicket Keeper", basePrice:10000000 },
  { name:"PRABHSIMRAN SINGH", country:"India", role:"Wicket Keeper", basePrice:10000000 },
  { name:"ISHAN KISHAN", country:"India", role:"Wicket Keeper", basePrice:15000000 },
  { name:"VISHNU VINOD", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"PHILIP SALT", country:"England", role:"Wicket Keeper", basePrice:15000000 },
  { name:"SAM BILLINGS", country:"England", role:"Wicket Keeper", basePrice:10000000 },
  { name:"JOSHUA PHILIPPE", country:"Australia", role:"Wicket Keeper", basePrice:10000000 },
  { name:"RISHAB PANT", country:"India", role:"Wicket Keeper", basePrice:20000000 },
  { name:"AKSHDEEP NATH", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"JOSH INGLIS", country:"Australia", role:"Wicket Keeper", basePrice:10000000 },
  { name:"DHRUV JUREL", country:"India", role:"Wicket Keeper", basePrice:10000000 },
  { name:"VAIBHAV SURYAVANSHI", country:"India", role:"Batter", basePrice:5000000 },
  { name:"ZEESHAN ANSARI", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"JACOB BETHELL", country:"England", role:"All-Rounder", basePrice:5000000 },
  { name:"AYUSH MATHRE", country:"India", role:"Batter", basePrice:10000000 },
  { name:"ABHINAV MANOHAR", country:"India", role:"Batter", basePrice:2000000 },
  { name:"NAMAN DHIR", country:"India", role:"All-Rounder", basePrice:10000000 },
  { name:"ARJUN TENDULKAR", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"DIGVESH RATHI", country:"India", role:"Bowler", basePrice:5000000 },
  { name:"PRIYANSH ARYA", country:"India", role:"Batter", basePrice:5000000 },
  { name:"VIPRAJ NIGAM", country:"India", role:"All-Rounder", basePrice:5000000 },
  { name:"BRYDON CARSE", country:"England", role:"All-Rounder", basePrice:10000000 },
  { name:"XAVIER BARTLETT", country:"Australia", role:"Bowler", basePrice:5000000 },
  { name:"ESHAN MALINGA", country:"Sri Lanka", role:"Bowler", basePrice:5000000 },
  { name:"SAURABH NETRAVALKAR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"VIGNESH PUTHUR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"ANSHUL KAMBOJ", country:"India", role:"Bowler", basePrice:10000000 },
  { name:"KAMINDU MENDIS", country:"Sri Lanka", role:"All-Rounder", basePrice:5000000 },
  { name:"PRINCE YADAV", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"ADAM MILNE", country:"New Zealand", role:"Bowler", basePrice:5000000 },
  { name:"SATYANARAYANA RAJU", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"WAYNE PARNELL", country:"South Africa", role:"All-Rounder", basePrice:5000000 },
  { name:"GURJAPNEET SINGH", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"URVIL PATEL", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"JAMIE OVERTON", country:"England", role:"All-Rounder", basePrice:5000000 },
  { name:"TRISTAN STUBBS", country:"South Africa", role:"Wicket Keeper", basePrice:15000000 },
  { name:"KUSAL MENDIS", country:"Sri Lanka", role:"Batter", basePrice:10000000 },
  { name:"SAI SUDARSHAN", country:"India", role:"Batter", basePrice:10000000 },
  { name:"AYUSH BADONI", country:"India", role:"Batter", basePrice:5000000 },
  { name:"SARFARAZ KHAN", country:"India", role:"Batter", basePrice:2000000 },
  { name:"KARTIK SHARMA", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"RAMAKRISHNA GHOSH", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"PRASHANT VEER", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"AKEAL HOSEIN", country:"West Indies", role:"Bowler", basePrice:5000000 },
  { name:"AUQIB NABI", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"FINN ALLEN", country:"New Zealand", role:"Wicket Keeper", basePrice:10000000 },
  { name:"TIM SEIFERT", country:"New Zealand", role:"Wicket Keeper", basePrice:5000000 },
  { name:"BLESSING MUZARABANI", country:"Zimbabwe", role:"Bowler", basePrice:2000000 },
  { name:"MUKUL CHOUDHARY", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"RYAN RICKELTON", country:"South Africa", role:"Wicket Keeper", basePrice:10000000 },
  { name:"RAJ ANGAD BAWA", country:"India", role:"All-Rounder", basePrice:2000000 },
  { name:"CORBIN BOSCH", country:"South Africa", role:"All-Rounder", basePrice:5000000 },
  { name:"ALLAH GHAZANFAR", country:"Afghanistan", role:"Bowler", basePrice:2000000 },
  { name:"COOPER CONNOLLY", country:"Australia", role:"All-Rounder", basePrice:10000000 },
  { name:"DONOVAN FERREIRA", country:"South Africa", role:"Wicket Keeper", basePrice:10000000 },
  { name:"BRIJESH SHARMA", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"JORDAN COX", country:"England", role:"Wicket Keeper", basePrice:2000000 },
  { name:"MANGESH YADAV", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"JACOB DUFFY", country:"New Zealand", role:"Bowler", basePrice:10000000 },
  { name:"SALIL ARORA", country:"India", role:"Wicket Keeper", basePrice:2000000 },
  { name:"SHIVANG KUMAR", country:"India", role:"Bowler", basePrice:2000000 },
  { name:"PRAFUL HINGE", country:"India", role:"Bowler", basePrice:2000000 }
];

let currentRole = "All";

let ascending = true;

const tableBody =
    document.getElementById("playersTable");

function formatPrice(price){

    return "₹ " + (price/10000000).toFixed(2) + " Cr";
}

function renderPlayers(){

    const search =
        document.getElementById("searchInput")
        .value
        .toLowerCase();

    let filtered =
        players.filter(player => {

            const matchesSearch =
                player.name
                .toLowerCase()
                .includes(search);

            const matchesRole =
                currentRole === "All"
                ||
                player.role === currentRole;

            return matchesSearch
                &&
                matchesRole;
        });

    tableBody.innerHTML = "";

    filtered.forEach(player => {

        tableBody.innerHTML += `

            <tr>

                <td data-label="Name">${player.name}</td>

                <td data-label="Country">${player.country}</td>

                <td data-label="Role">
                    <span class="role-badge">
                        ${player.role}
                    </span>
                </td>

                <td class="price" data-label="Base Price">
                    ${formatPrice(player.basePrice)}
                </td>

            </tr>

        `;
    });
}

renderPlayers();

/* SEARCH */

document
.getElementById("searchInput")
.addEventListener("input",renderPlayers);

/* FILTER */

document
.querySelectorAll(".filter-btn")
.forEach(btn => {

    btn.addEventListener("click",() => {

        document
        .querySelectorAll(".filter-btn")
        .forEach(b => b.classList.remove("active"));

        btn.classList.add("active");

        currentRole =
            btn.dataset.role;

        renderPlayers();
    });

});

/* SORT */

document
.getElementById("sortBtn")
.addEventListener("click",() => {

    players.sort((a,b)=>{

        return ascending
            ?
            a.basePrice-b.basePrice
            :
            b.basePrice-a.basePrice;
    });

    ascending = !ascending;

    renderPlayers();
});