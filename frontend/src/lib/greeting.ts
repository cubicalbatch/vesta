export function timeOfDayGreeting(date = new Date()): string {
	const h = date.getHours();
	if (h < 12) return 'Good morning';
	if (h < 18) return 'Good afternoon';
	return 'Good evening';
}
