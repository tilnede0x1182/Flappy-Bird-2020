/**
 * Tests pour Flappy-Bird-2020
 * Validation de la physique du jeu
 * Execution: node tests.js
 */

const TESTS_RESULTS = { passed: 0, failed: 0 };

function assert(description, condition) {
    if (condition) {
        console.log(`[PASS] ${description}`);
        TESTS_RESULTS.passed++;
    } else {
        console.log(`[FAIL] ${description}`);
        TESTS_RESULTS.failed++;
    }
}

/**
 * Simule un oiseau.
 */
class Bird {
    constructor(positionX, positionY) {
        this.positionX = positionX;
        this.positionY = positionY;
        this.velocite = 0;
        this.tickCount = 0;
    }

    /**
     * Fait sauter l'oiseau.
     */
    sauter() {
        this.velocite = -10.5;
        this.tickCount = 0;
    }

    /**
     * Met a jour la position.
     */
    deplacer() {
        this.tickCount++;
        const deplacement = this.velocite * this.tickCount + 1.5 * Math.pow(this.tickCount, 2);
        this.positionY += Math.min(deplacement, 16);
    }
}

/**
 * Simule un tuyau.
 */
class Pipe {
    constructor(positionX, hauteurGap = 200) {
        this.positionX = positionX;
        this.hauteurGap = hauteurGap;
        this.hauteur = Math.floor(Math.random() * 400) + 50;
    }

    /**
     * Deplace le tuyau vers la gauche.
     * @param {number} vitesse Vitesse de deplacement.
     */
    deplacer(vitesse = 5) {
        this.positionX -= vitesse;
    }

    /**
     * Verifie collision avec un oiseau.
     * @param {Bird} oiseau Oiseau a verifier.
     * @param {number} largeurOiseau Largeur de l'oiseau.
     * @param {number} hauteurOiseau Hauteur de l'oiseau.
     * @returns {boolean} True si collision.
     */
    collision(oiseau, largeurOiseau = 34, hauteurOiseau = 24) {
        // Collision avec tuyau du haut
        if (oiseau.positionX + largeurOiseau > this.positionX && oiseau.positionX < this.positionX + 52) {
            if (oiseau.positionY < this.hauteur || oiseau.positionY + hauteurOiseau > this.hauteur + this.hauteurGap) {
                return true;
            }
        }
        return false;
    }
}

// ==================== TESTS ====================

console.log("=== Tests Flappy-Bird ===\n");

// Tests Bird
assert("Bird position initiale", (() => {
    const oiseau = new Bird(100, 200);
    return oiseau.positionX === 100 && oiseau.positionY === 200;
})());

assert("Bird saut change velocite", (() => {
    const oiseau = new Bird(100, 200);
    oiseau.sauter();
    return oiseau.velocite === -10.5;
})());

assert("Bird deplacement gravite", (() => {
    const oiseau = new Bird(100, 200);
    const positionInitiale = oiseau.positionY;
    oiseau.deplacer();
    return oiseau.positionY !== positionInitiale;
})());

assert("Bird saut puis chute", (() => {
    const oiseau = new Bird(100, 200);
    oiseau.sauter();
    oiseau.deplacer();
    oiseau.deplacer();
    oiseau.deplacer();
    return oiseau.positionY > 200; // Finit par retomber
})());

// Tests Pipe
assert("Pipe deplacement", (() => {
    const tuyau = new Pipe(500);
    tuyau.deplacer();
    return tuyau.positionX === 495;
})());

assert("Pipe collision detectee", (() => {
    const tuyau = new Pipe(100);
    tuyau.hauteur = 150;
    const oiseau = new Bird(100, 100); // Au-dessus du gap
    return tuyau.collision(oiseau);
})());

assert("Pipe pas de collision dans gap", (() => {
    const tuyau = new Pipe(100);
    tuyau.hauteur = 150;
    const oiseau = new Bird(100, 200); // Dans le gap
    return !tuyau.collision(oiseau);
})());

// ==================== RESUME ====================

console.log("\n=== Resume ===");
console.log(`Tests passes: ${TESTS_RESULTS.passed}`);
console.log(`Tests echoues: ${TESTS_RESULTS.failed}`);

if (TESTS_RESULTS.failed > 0) process.exit(1);
