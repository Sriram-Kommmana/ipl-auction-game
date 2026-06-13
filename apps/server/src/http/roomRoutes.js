import express from 'express';
const router = express.Router();

router.post('/create', (req, res) => res.json({ message: 'ok' }))
router.post('/join', (req, res) => res.json({ message: 'ok' }))
router.get('/:roomId', (req, res) => res.json({ message: 'ok' }))

export default router;