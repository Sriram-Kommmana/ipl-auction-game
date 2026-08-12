const transition =
    document.querySelector(".page-transition");

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

    {
        name:"Virat Kohli",
        country:"India",
        role:"Batter",
        basePrice:20000000
    },

    {
        name:"Jasprit Bumrah",
        country:"India",
        role:"Bowler",
        basePrice:20000000
    },

    {
        name:"Hardik Pandya",
        country:"India",
        role:"All-Rounder",
        basePrice:20000000
    },

    {
        name:"Rohit Sharma",
        country:"India",
        role:"Batter",
        basePrice:20000000
    },

    {
        name:"Jos Buttler",
        country:"England",
        role:"Wicket Keeper",
        basePrice:20000000
    },

    {
        name:"Travis Head",
        country:"Australia",
        role:"Batter",
        basePrice:20000000
    }

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

                <td>${player.name}</td>

                <td>${player.country}</td>

                <td>
                    <span class="role-badge">
                        ${player.role}
                    </span>
                </td>

                <td class="price">
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